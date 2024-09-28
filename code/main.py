import os
from PIL import Image
import numpy as np
import math
from scipy import spatial
from scipy.spatial import cKDTree, distance
import scipy.ndimage as ndi


def check_nD(array, ndim, arg_name="image"):
    array = np.asanyarray(array)
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, "-or-".join([str(n) for n in ndim]))
        )


def img_as_float(image, force_copy=False):
    return _convert(image, np.floating, force_copy)


def _dtype_bits(kind, bits, itemsize=1):
    s = next(
        i
        for i in (itemsize,) + (2, 4, 8)
        if bits < (i * 8) or (bits == (i * 8) and kind == "u")
    )

    return np.dtype(kind + str(s))


def _scale(a, n, m, copy=True):

    kind = a.dtype.kind
    if n > m and a.max() < 2**m:
        mnew = int(np.ceil(m / 2) * 2)
        if mnew > m:
            dtype = f"int{mnew}"
        else:
            dtype = f"uint{mnew}"
        n = int(np.ceil(n / 2) * 2)
        return a.astype(_dtype_bits(kind, m))
    elif n == m:
        return a.copy() if copy else a
    elif n > m:
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.floor_divide(a, 2 ** (n - m), out=b, dtype=a.dtype, casting="unsafe")
            return b
        else:
            a //= 2 ** (n - m)
            return a
    elif m % n == 0:
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, m))
            np.multiply(a, (2**m - 1) // (2**n - 1), out=b, dtype=b.dtype)
            return b
        else:
            a = a.astype(_dtype_bits(kind, m, a.dtype.itemsize), copy=False)
            a *= (2**m - 1) // (2**n - 1)
            return a
    else:
        o = (m // n + 1) * n
        if copy:
            b = np.empty(a.shape, _dtype_bits(kind, o))
            np.multiply(a, (2**o - 1) // (2**n - 1), out=b, dtype=b.dtype)
            b //= 2 ** (o - m)
            return b
        else:
            a = a.astype(_dtype_bits(kind, o, a.dtype.itemsize), copy=False)
            a *= (2**o - 1) // (2**n - 1)
            a //= 2 ** (o - m)
            return a


def _dtype_itemsize(itemsize, *dtypes):
    return next(dt for dt in dtypes if np.dtype(dt).itemsize >= itemsize)


def _convert(image, dtype, force_copy=False, uniform=False):
    dtype_range = {
        bool: (False, True),
        np.bool_: (False, True),
        float: (-1, 1),
        np.float16: (-1, 1),
        np.float32: (-1, 1),
        np.float64: (-1, 1),
    }
    _supported_types = list(dtype_range.keys())

    image = np.asarray(image)
    dtypeobj_in = image.dtype
    if dtype is np.floating:
        dtypeobj_out = np.dtype("float64")
    else:
        dtypeobj_out = np.dtype(dtype)
    dtype_in = dtypeobj_in.type
    dtype_out = dtypeobj_out.type
    kind_in = dtypeobj_in.kind
    kind_out = dtypeobj_out.kind
    itemsize_in = dtypeobj_in.itemsize
    itemsize_out = dtypeobj_out.itemsize

    if np.issubdtype(dtype_in, dtype):
        if force_copy:
            image = image.copy()
        return image

    if not (dtype_in in _supported_types and dtype_out in _supported_types):
        raise ValueError(f"Cannot convert from {dtypeobj_in} to " f"{dtypeobj_out}.")

    if kind_in in "ui":
        imin_in = np.iinfo(dtype_in).min
        imax_in = np.iinfo(dtype_in).max
    if kind_out in "ui":
        imin_out = np.iinfo(dtype_out).min
        imax_out = np.iinfo(dtype_out).max

    if kind_out == "b":
        return image > dtype_in(dtype_range[dtype_in][1] / 2)

    if kind_in == "b":
        result = image.astype(dtype_out)
        if kind_out != "f":
            result *= dtype_out(dtype_range[dtype_out][1])
        return result

    if kind_in == "f":
        if kind_out == "f":
            return image.astype(dtype_out)

        if np.min(image) < -1.0 or np.max(image) > 1.0:
            raise ValueError("Images of type float must be between -1 and 1.")
        computation_type = _dtype_itemsize(
            itemsize_out, dtype_in, np.float32, np.float64
        )

        if not uniform:
            if kind_out == "u":
                image_out = np.multiply(image, imax_out, dtype=computation_type)
            else:
                image_out = np.multiply(
                    image, (imax_out - imin_out) / 2, dtype=computation_type
                )
                image_out -= 1.0 / 2.0
            np.rint(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        elif kind_out == "u":
            image_out = np.multiply(image, imax_out + 1, dtype=computation_type)
            np.clip(image_out, 0, imax_out, out=image_out)
        else:
            image_out = np.multiply(
                image, (imax_out - imin_out + 1.0) / 2.0, dtype=computation_type
            )
            np.floor(image_out, out=image_out)
            np.clip(image_out, imin_out, imax_out, out=image_out)
        return image_out.astype(dtype_out)

    # signed/unsigned int -> float
    if kind_out == "f":
        # use float type that can exactly represent input integers
        computation_type = _dtype_itemsize(
            itemsize_in, dtype_out, np.float32, np.float64
        )

        if kind_in == "u":
            # using np.divide or np.multiply doesn't copy the data
            # until the computation time
            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
            # DirectX uses this conversion also for signed ints
            # if imin_in:
            #     np.maximum(image, -1.0, out=image)
        elif kind_in == "i":
            # From DirectX conversions:
            # The most negative value maps to -1.0f
            # Every other value is converted to a float (call it c)
            # and then result = c * (1.0f / (2⁽ⁿ⁻¹⁾-1)).

            image = np.multiply(image, 1.0 / imax_in, dtype=computation_type)
            np.maximum(image, -1.0, out=image)

        else:
            image = np.add(image, 0.5, dtype=computation_type)
            image *= 2 / (imax_in - imin_in)

        return np.asarray(image, dtype_out)

    # unsigned int -> signed/unsigned int
    if kind_in == "u":
        if kind_out == "i":
            # unsigned int -> signed int
            image = _scale(image, 8 * itemsize_in, 8 * itemsize_out - 1)
            return image.view(dtype_out)
        else:
            # unsigned int -> unsigned int
            return _scale(image, 8 * itemsize_in, 8 * itemsize_out)

    # signed int -> unsigned int
    if kind_out == "u":
        image = _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out)
        result = np.empty(image.shape, dtype_out)
        np.maximum(image, 0, out=result, dtype=image.dtype, casting="unsafe")
        return result

    # signed int -> signed int
    if itemsize_in > itemsize_out:
        return _scale(image, 8 * itemsize_in - 1, 8 * itemsize_out - 1)

    image = image.astype(_dtype_bits("i", itemsize_out * 8))
    image -= imin_in
    image = _scale(image, 8 * itemsize_in, 8 * itemsize_out, copy=False)
    image += imin_out
    return image.astype(dtype_out)


def _supported_float_type(input_dtype, allow_complex=False):
    new_float_type = {
        # preserved types
        np.float32().dtype.char: np.float32,
        np.float64().dtype.char: np.float64,
        np.complex64().dtype.char: np.complex64,
        np.complex128().dtype.char: np.complex128,
        # altered types
        np.float16().dtype.char: np.float32,
        "g": np.float64,  # np.float128 ; doesn't exist on windows
        "G": np.complex128,  # np.complex256 ; doesn't exist on windows
    }

    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == "c":
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def integral_image(image, *, dtype=None):
    if dtype is None and image.real.dtype.kind == "f":
        dtype = np.promote_types(image.dtype, np.float64)

    S = image
    for i in range(image.ndim):
        S = S.cumsum(axis=i, dtype=dtype)
    return S


def _hessian_matrix_det(img, sigma):
    def _integ(img, r, c, rl, cl):
        def _clip(x, low, high):
            assert 0 <= low <= high

            if x > high:
                return high
            elif x < low:
                return low
            else:
                return x

        r = _clip(r, 0, img.shape[0] - 1)
        c = _clip(c, 0, img.shape[1] - 1)

        r2 = _clip(r + rl, 0, img.shape[0] - 1)
        c2 = _clip(c + cl, 0, img.shape[1] - 1)

        ans = img[r, c] + img[r2, c2] - img[r, c2] - img[r2, c]
        return max(0.0, ans)

    size = int(3 * sigma)
    height, width = img.shape
    s2 = (size - 1) // 2
    s3 = size // 3
    w = size
    out = np.empty_like(img, dtype=np.float64)
    w_i = 1.0 / size / size

    if size % 2 == 0:
        size += 1

    for r in range(height):
        for c in range(width):
            tl = _integ(img, r - s3, c - s3, s3, s3)  # top left
            br = _integ(img, r + 1, c + 1, s3, s3)  # bottom right
            bl = _integ(img, r - s3, c + 1, s3, s3)  # bottom left
            tr = _integ(img, r + 1, c - s3, s3, s3)  # top right

            dxy = bl + tr - tl - br
            dxy = -dxy * w_i

            mid = _integ(img, r - s3 + 1, c - s2, 2 * s3 - 1, w)  # middle box
            side = _integ(img, r - s3 + 1, c - s3 // 2, 2 * s3 - 1, s3)  # sides

            dxx = mid - 3 * side
            dxx = -dxx * w_i

            mid = _integ(img, r - s2, c - s3 + 1, w, 2 * s3 - 1)
            side = _integ(img, r - s3 // 2, c - s3 + 1, s3, 2 * s3 - 1)

            dyy = mid - 3 * side
            dyy = -dyy * w_i

            out[r, c] = dxx * dyy - 0.81 * (dxy * dxy)

    return out


def peak_local_max(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):
    def _get_peak_mask(image, footprint, threshold, mask=None):
        if footprint.size == 1 or image.size == 1:
            return image > threshold

        image_max = ndi.maximum_filter(image, footprint=footprint, mode="nearest")

        out = image == image_max

        # no peak for a trivial image
        image_is_trivial = np.all(out) if mask is None else np.all(out[mask])
        if image_is_trivial:
            out[:] = False
            if mask is not None:
                # isolated pixels in masked area are returned as peaks
                isolated_px = np.logical_xor(mask, ndi.binary_opening(mask))
                out[isolated_px] = True

        out &= image > threshold
        return out

    def _get_threshold(image, threshold_abs, threshold_rel):
        threshold = threshold_abs if threshold_abs is not None else image.min()

        if threshold_rel is not None:
            threshold = max(threshold, threshold_rel * image.max())

        return threshold

    def _exclude_border(label, border_width):
        for i, width in enumerate(border_width):
            if width == 0:
                continue
            label[(slice(None),) * i + (slice(None, width),)] = 0
            label[(slice(None),) * i + (slice(-width, None),)] = 0
        return label

    def _get_excluded_border_width(image, min_distance, exclude_border):
        if isinstance(exclude_border, bool):
            border_width = (min_distance if exclude_border else 0,) * image.ndim
        elif isinstance(exclude_border, int):
            if exclude_border < 0:
                raise ValueError("`exclude_border` cannot be a negative value")
            border_width = (exclude_border,) * image.ndim
        elif isinstance(exclude_border, tuple):
            if len(exclude_border) != image.ndim:
                raise ValueError(
                    "`exclude_border` should have the same length as the "
                    "dimensionality of the image."
                )
            for exclude in exclude_border:
                if not isinstance(exclude, int):
                    raise ValueError(
                        "`exclude_border`, when expressed as a tuple, must only "
                        "contain ints."
                    )
                if exclude < 0:
                    raise ValueError("`exclude_border` can not be a negative value")
            border_width = exclude_border
        else:
            raise TypeError(
                "`exclude_border` must be bool, int, or tuple with the same "
                "length as the dimensionality of the image."
            )

        return border_width

    def _get_high_intensity_peaks(image, mask, num_peaks, min_distance, p_norm):

        def ensure_spacing(
            coords,
            spacing=1,
            p_norm=np.inf,
            min_split_size=50,
            max_out=None,
            *,
            max_split_size=2000,
        ):

            def _ensure_spacing(coord, spacing, p_norm, max_out):
                # Use KDtree to find the peaks that are too close to each other
                tree = cKDTree(coord)

                indices = tree.query_ball_point(coord, r=spacing, p=p_norm)
                rejected_peaks_indices = set()
                naccepted = 0
                for idx, candidates in enumerate(indices):
                    if idx not in rejected_peaks_indices:
                        # keep current point and the points at exactly spacing from it
                        candidates.remove(idx)
                        dist = distance.cdist(
                            [coord[idx]],
                            coord[candidates],
                            distance.minkowski,
                            p=p_norm,
                        ).reshape(-1)
                        candidates = [
                            c for c, d in zip(candidates, dist) if d < spacing
                        ]

                        # candidates.remove(keep)
                        rejected_peaks_indices.update(candidates)
                        naccepted += 1
                        if max_out is not None and naccepted >= max_out:
                            break

                # Remove the peaks that are too close to each other
                output = np.delete(coord, tuple(rejected_peaks_indices), axis=0)
                if max_out is not None:
                    output = output[:max_out]

                return output

            output = coords
            if len(coords):
                coords = np.atleast_2d(coords)
                if min_split_size is None:
                    batch_list = [coords]
                else:
                    coord_count = len(coords)
                    split_idx = [min_split_size]
                    split_size = min_split_size
                    while coord_count - split_idx[-1] > max_split_size:
                        split_size *= 2
                        split_idx.append(
                            split_idx[-1] + min(split_size, max_split_size)
                        )
                    batch_list = np.array_split(coords, split_idx)

                output = np.zeros((0, coords.shape[1]), dtype=coords.dtype)
                for batch in batch_list:
                    output = _ensure_spacing(
                        np.vstack([output, batch]), spacing, p_norm, max_out
                    )
                    if max_out is not None and len(output) >= max_out:
                        break

            return output

        coord = np.nonzero(mask)
        intensities = image[coord]
        idx_maxsort = np.argsort(-intensities, kind="stable")
        coord = np.transpose(coord)[idx_maxsort]

        if np.isfinite(num_peaks):
            max_out = int(num_peaks)
        else:
            max_out = None

        coord = ensure_spacing(
            coord, spacing=min_distance, p_norm=p_norm, max_out=max_out
        )

        if len(coord) > num_peaks:
            coord = coord[:num_peaks]

        return coord

    border_width = _get_excluded_border_width(image, min_distance, exclude_border)

    threshold = _get_threshold(image, threshold_abs, threshold_rel)

    if footprint is None:
        size = 2 * min_distance + 1
        footprint = np.ones((size,) * image.ndim, dtype=bool)
    else:
        footprint = np.asarray(footprint)

    if labels is None:
        # Non maximum filter
        mask = _get_peak_mask(image, footprint, threshold)

        mask = _exclude_border(mask, border_width)

        # Select highest intensities (num_peaks)
        coordinates = _get_high_intensity_peaks(
            image, mask, num_peaks, min_distance, p_norm
        )

    else:
        _labels = _exclude_border(labels.astype(int, casting="safe"), border_width)

        if np.issubdtype(image.dtype, np.floating):
            bg_val = np.finfo(image.dtype).min
        else:
            bg_val = np.iinfo(image.dtype).min

        # For each label, extract a smaller image enclosing the object of
        # interest, identify num_peaks_per_label peaks
        labels_peak_coord = []

        for label_idx, roi in enumerate(ndi.find_objects(_labels)):
            if roi is None:
                continue

            # Get roi mask
            label_mask = labels[roi] == label_idx + 1
            # Extract image roi
            img_object = image[roi].copy()
            # Ensure masked values don't affect roi's local peaks
            img_object[np.logical_not(label_mask)] = bg_val

            mask = _get_peak_mask(img_object, footprint, threshold, label_mask)

            coordinates = _get_high_intensity_peaks(
                img_object, mask, num_peaks_per_label, min_distance, p_norm
            )

            # transform coordinates in global image indices space
            for idx, s in enumerate(roi):
                coordinates[:, idx] += s.start

            labels_peak_coord.append(coordinates)

        if labels_peak_coord:
            coordinates = np.vstack(labels_peak_coord)
        else:
            coordinates = np.empty((0, 2), dtype=int)

        if len(coordinates) > num_peaks:
            out = np.zeros_like(image, dtype=bool)
            out[tuple(coordinates.T)] = True
            coordinates = _get_high_intensity_peaks(
                image, out, num_peaks, min_distance, p_norm
            )

    return coordinates


def _prune_blobs(blobs_array, overlap, *, sigma_dim=1):
    def _blob_overlap(blob1, blob2, *, sigma_dim=1):
        def _compute_disk_overlap(d, r1, r2):
            ratio1 = (d**2 + r1**2 - r2**2) / (2 * d * r1)
            ratio1 = np.clip(ratio1, -1, 1)
            acos1 = math.acos(ratio1)

            ratio2 = (d**2 + r2**2 - r1**2) / (2 * d * r2)
            ratio2 = np.clip(ratio2, -1, 1)
            acos2 = math.acos(ratio2)

            a = -d + r2 + r1
            b = d - r2 + r1
            c = d + r2 - r1
            d = d + r2 + r1
            area = r1**2 * acos1 + r2**2 * acos2 - 0.5 * math.sqrt(abs(a * b * c * d))
            return area / (math.pi * (min(r1, r2) ** 2))

        def _compute_sphere_overlap(d, r1, r2):
            vol = (
                math.pi
                / (12 * d)
                * (r1 + r2 - d) ** 2
                * (d**2 + 2 * d * (r1 + r2) - 3 * (r1**2 + r2**2) + 6 * r1 * r2)
            )
            return vol / (4.0 / 3 * math.pi * min(r1, r2) ** 3)

        ndim = len(blob1) - sigma_dim
        if ndim > 3:
            return 0.0
        root_ndim = math.sqrt(ndim)

        if blob1[-1] == blob2[-1] == 0:
            return 0.0
        elif blob1[-1] > blob2[-1]:
            max_sigma = blob1[-sigma_dim:]
            r1 = 1
            r2 = blob2[-1] / blob1[-1]
        else:
            max_sigma = blob2[-sigma_dim:]
            r2 = 1
            r1 = blob1[-1] / blob2[-1]
        pos1 = blob1[:ndim] / (max_sigma * root_ndim)
        pos2 = blob2[:ndim] / (max_sigma * root_ndim)

        d = np.sqrt(np.sum((pos2 - pos1) ** 2))
        if d > r1 + r2:  # centers farther than sum of radii, so no overlap
            return 0.0

        # one blob is inside the other
        if d <= abs(r1 - r2):
            return 1.0

        if ndim == 2:
            return _compute_disk_overlap(d, r1, r2)

        else:  # ndim=3 http://mathworld.wolfram.com/Sphere-SphereIntersection.html
            return _compute_sphere_overlap(d, r1, r2)

    sigma = blobs_array[:, -sigma_dim:].max()
    distance = 2 * sigma * math.sqrt(blobs_array.shape[1] - sigma_dim)
    tree = spatial.cKDTree(blobs_array[:, :-sigma_dim])
    pairs = np.array(list(tree.query_pairs(distance)))
    if len(pairs) == 0:
        return blobs_array
    else:
        for i, j in pairs:
            blob1, blob2 = blobs_array[i], blobs_array[j]
            if _blob_overlap(blob1, blob2, sigma_dim=sigma_dim) > overlap:
                # note: this test works even in the anisotropic case because
                # all sigmas increase together.
                if blob1[-1] > blob2[-1]:
                    blob2[-1] = 0
                else:
                    blob1[-1] = 0

    return np.stack([b for b in blobs_array if b[-1] > 0])


def blob_doh(
    image,
    min_sigma=1,
    max_sigma=30,
    num_sigma=10,
    threshold=0.01,
    overlap=0.5,
    log_scale=False,
    *,
    threshold_rel=None,
):
    check_nD(image, 2)

    image = img_as_float(image)
    float_dtype = _supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    image = integral_image(image)

    if log_scale:
        start, stop = math.log(min_sigma, 10), math.log(max_sigma, 10)
        sigma_list = np.logspace(start, stop, num_sigma)
    else:
        sigma_list = np.linspace(min_sigma, max_sigma, num_sigma)

    image_cube = np.empty(shape=image.shape + (len(sigma_list),), dtype=float_dtype)
    for j, s in enumerate(sigma_list):
        image_cube[..., j] = _hessian_matrix_det(image, s)

    local_maxima = peak_local_max(
        image_cube,
        threshold_abs=threshold,
        threshold_rel=threshold_rel,
        exclude_border=False,
        footprint=np.ones((3,) * image_cube.ndim),
    )

    # Catch no peaks
    if local_maxima.size == 0:
        return np.empty((0, 3))
    # Convert local_maxima to float64
    lm = local_maxima.astype(np.float64)
    # Convert the last index to its corresponding scale value
    lm[:, -1] = sigma_list[local_maxima[:, -1]]
    return _prune_blobs(lm, overlap)


def get_x(pos: float) -> str | None:
    if 45.0 <= pos <= 52.0:
        return "A"
    elif 178.0 <= pos <= 186.0:
        return "B"
    elif 310.0 <= pos <= 320.0:
        return "C"
    elif 445.0 <= pos <= 453.0:
        return "D"
    else:
        return None


def get_y(pos: float) -> int | None:
    if 45.0 <= pos <= 52.0:
        return 0
    elif 178.0 <= pos <= 186.0:
        return 1
    elif 310.0 <= pos <= 320.0:
        return 2
    elif 445.0 <= pos <= 453.0:
        return 3
    else:
        return None


file = input()
answer = input().split()

image = np.array(Image.open(file), dtype=np.float32)
blobs = blob_doh(image, max_sigma=20, threshold=0.01)

answers = ["", "", "", ""]

for idx, blob in enumerate(blobs):
    y, x, r = blob

    if r > 14 and r < 20 and len(answers) <= 4:
        img = Image.open(file)

        if (idx := get_y(y)) is not None:
            opt = get_x(x)
            if opt is not None:
                answers[idx] = opt

        bbox = (x - r, y - r, x + r, y + r)
        output = img.crop(bbox)

count = 0
for idx, x in enumerate(answers):
    if x == answer[idx]:
        count += 1

print(count)
