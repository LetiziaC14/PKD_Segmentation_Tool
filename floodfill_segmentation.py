import argparse
import os
import cv2
import numpy as np


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class FloodFillSegmenter:
    def __init__(self, img_path, out_dir, tol=10, radius=20):
        self.img_path = img_path
        self.out_dir = out_dir
        self.tol = int(tol)
        self.radius = int(radius)
        self.img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.img_color is None:
            raise ValueError(f"Cannot read image: {img_path}")
        self.img = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.h, self.w = self.img.shape[:2]
        self.masks = []  # list of binary masks (h,w) uint8
        self.mask_colors = []  # list of (B,G,R) colors for each mask
        self.preview = self.img_color.copy()
        # manual editing
        self.manual_mode = False
        self.painting = False
        self.paint_erase = False
        self.brush = 5
        self.last_seed = None

    def set_tolerance(self, tol):
        self.tol = int(tol)

    def set_radius(self, r):
        self.radius = int(r)

    def apply_seed(self, x, y):
        # create mask of pixels whose intensity is within tol of the seed intensity
        sx = int(x)
        sy = int(y)
        if sx < 0 or sx >= self.w or sy < 0 or sy >= self.h:
            return None
        seed_val = int(self.img[sy, sx])
        # intensity condition
        int_mask = np.abs(self.img.astype(int) - seed_val) <= self.tol
        # spatial condition (Euclidean distance)
        yy, xx = np.ogrid[:self.h, :self.w]
        dist2 = (xx - sx) ** 2 + (yy - sy) ** 2
        spa_mask = dist2 <= (self.radius ** 2)
        candidate = np.logical_and(int_mask, spa_mask).astype(np.uint8)
        # keep only connected component containing the seed
        if candidate[sy, sx] == 0:
            # record attempted seed for debugging marker
            self.last_seed = (sx, sy)
            return None
        num, labels = cv2.connectedComponents(candidate)
        label_seed = labels[sy, sx]
        mask_used = (labels == label_seed).astype(np.uint8) * 255
        # fill internal holes so segmentation is a solid surface
        mask_used = self._fill_holes(mask_used)
        self.masks.append(mask_used)
        color = tuple(int(c) for c in np.random.randint(0, 256, size=3))
        self.mask_colors.append(color)
        self.last_seed = (sx, sy)
        return mask_used

    def undo(self):
        if self.masks:
            self.masks.pop()
            self.mask_colors.pop()
            return True
        return False

    def _fill_holes(self, mask):
        # mask: uint8 0/255
        h, w = mask.shape
        # morphological closing to remove small holes/gaps
        k = max(3, min(51, (self.radius // 10) * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # fill internal holes using floodFill on the inverted image
        inv = cv2.bitwise_not(closed)
        im_floodfill = inv.copy()
        mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(im_floodfill, mask_ff, (0, 0), 255)
        holes = cv2.bitwise_not(im_floodfill)
        filled = cv2.bitwise_or(closed, holes)
        return filled

    def save_masks(self, save_binaries=True):
        base = os.path.splitext(os.path.basename(self.img_path))[0]
        ensure_dir(self.out_dir)
        saved = []
        # ensure masks are hole-filled before saving
        masks_to_save = [self._fill_holes(m) for m in self.masks]
        if save_binaries:
            for i, m in enumerate(masks_to_save, start=1):
                fname = f"{base}_mask_{i}.png"
                out_path = os.path.join(self.out_dir, fname)
                cv2.imwrite(out_path, m)
                saved.append(out_path)
        # also save a composite colored overlay image if any masks exist
        if self.masks:
            overlay = self.img_color.copy()
            alpha = 0.5
            for m, color in zip(masks_to_save, self.mask_colors):
                mask_bool = m.astype(bool)
                color_img = np.full_like(overlay, color)
                blended = cv2.addWeighted(overlay, 1 - alpha, color_img, alpha, 0)
                overlay[mask_bool] = blended[mask_bool]
            overlay_name = f"{base}_masks_overlay.png"
            overlay_path = os.path.join(self.out_dir, overlay_name)
            cv2.imwrite(overlay_path, overlay)
            saved.append(overlay_path)
            # save a single class-colored mask image (each mask gets its color)
            class_img = np.zeros_like(self.img_color)
            for m, color in zip(masks_to_save, self.mask_colors):
                mask_bool = m.astype(bool)
                class_img[mask_bool] = color
            class_name = f"{base}_classmask.png"
            class_path = os.path.join(self.out_dir, class_name)
            cv2.imwrite(class_path, class_img)
            saved.append(class_path)
        return saved

    def overlay_preview(self, mode='both'):
        # create a colored overlay of all masks with distinct colors
        overlay = self.img_color.copy()
        alpha = 0.5
        for m, color in zip(self.masks, self.mask_colors):
            mask_bool = m.astype(bool)
            color_img = np.full_like(overlay, color)
            blended = cv2.addWeighted(overlay, 1 - alpha, color_img, alpha, 0)
            overlay[mask_bool] = blended[mask_bool]
        left = self.img_color.copy()
        if mode == 'both':
            return np.hstack((left, overlay))
        elif mode == 'overlay':
            return overlay
        else:
            return left


def list_images(folder):
    exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith(exts) and os.path.isfile(os.path.join(folder, f))]
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description='FloodFill interactive segmentation')
    parser.add_argument('--input', '-i', required=True, help='Input folder with images')
    parser.add_argument('--output', '-o', default='masks', help='Output folder for masks')
    parser.add_argument('--tol', '-t', type=int, default=10, help='FloodFill tolerance (lo/up)')
    parser.add_argument('--no-binaries', action='store_true', help='Do not save individual binary mask PNGs; save only composite/class-colored images')
    args = parser.parse_args()

    imgs = list_images(args.input)
    if not imgs:
        print('No images found in', args.input)
        return

    current_idx = 0

    print('Controls: left-click to add seed -> create mask;')
    print("'s' save masks for current image; 'r' reset masks; 'n' next image; 'p' prev image; 'q' quit")

    segmenter = FloodFillSegmenter(imgs[current_idx], os.path.join(args.input, args.output), tol=args.tol)

    window = 'FloodFillSeg'
    ctrl_win = 'Controls'
    # image window fixed size so mouse coords map 1:1
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    # separate controls window to avoid vertical offset inside image window
    cv2.namedWindow(ctrl_win, cv2.WINDOW_NORMAL)
    print('Image window fixed; controls are in separate window `Controls`. Click maps directly to image pixels.')

    display = {'mode': 'both', 'scale': 1}

    def on_mouse(event, x, y, flags, param):
        # map x/y from scaled display back to raw image coordinates
        scale = display['scale']
        img_x = x
        if display['mode'] == 'both':
            panel_width = segmenter.w * scale
            if img_x >= panel_width:
                img_x -= panel_width
        img_x = img_x // scale
        img_x = max(0, min(segmenter.w - 1, img_x))
        img_y = y // scale
        img_y = max(0, min(segmenter.h - 1, img_y))

        if segmenter.manual_mode:
            # painting mode: left button = add, right button = erase
            if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
                segmenter.painting = True
                segmenter.paint_erase = (event == cv2.EVENT_RBUTTONDOWN)
                # ensure there's a mask to edit
                if not segmenter.masks:
                    empty = np.zeros((segmenter.h, segmenter.w), dtype=np.uint8)
                    segmenter.masks.append(empty)
                    segmenter.mask_colors.append(tuple(int(c) for c in np.random.randint(0, 256, size=3)))
                val = 0 if segmenter.paint_erase else 255
                cv2.circle(segmenter.masks[-1], (img_x, img_y), segmenter.brush, val, -1)
            elif event == cv2.EVENT_MOUSEMOVE and segmenter.painting:
                val = 0 if segmenter.paint_erase else 255
                cv2.circle(segmenter.masks[-1], (img_x, img_y), segmenter.brush, val, -1)
            elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
                segmenter.painting = False
                segmenter.paint_erase = False
        else:
            # seed-based segmentation
            if event == cv2.EVENT_LBUTTONDOWN:
                mask = segmenter.apply_seed(img_x, img_y)
                if mask is None:
                    print(f'No mask created for seed=({img_x},{img_y}) with tol={segmenter.tol} radius={segmenter.radius}')
                else:
                    print(f'Added mask #{len(segmenter.masks)} seed=({img_x},{img_y}) tol={segmenter.tol} radius={segmenter.radius} -> mask pixels={mask.sum()//255}')

    cv2.setMouseCallback(window, on_mouse)

    # trackbars are kept in a separate Controls window to avoid shifting the image
    def on_trackbar_tol(val):
        segmenter.set_tolerance(val)

    def on_trackbar_radius(val):
        segmenter.set_radius(val)

    def on_trackbar_brush(val):
        segmenter.brush = max(1, val)

    def on_trackbar_zoom(val):
        display['scale'] = max(1, val)

    max_radius = max(segmenter.h, segmenter.w)
    cv2.createTrackbar('tol', ctrl_win, segmenter.tol, 255, on_trackbar_tol)
    cv2.createTrackbar('radius', ctrl_win, segmenter.radius, max_radius, on_trackbar_radius)
    cv2.createTrackbar('brush', ctrl_win, segmenter.brush, 100, on_trackbar_brush)
    cv2.createTrackbar('zoom', ctrl_win, display['scale'], 4, on_trackbar_zoom)

    while True:
        vis = segmenter.overlay_preview(mode=display['mode'])
        scale = display['scale']
        if scale > 1:
            display_vis = cv2.resize(vis, (vis.shape[1] * scale, vis.shape[0] * scale), interpolation=cv2.INTER_LINEAR)
        else:
            display_vis = vis
        # ensure window matches image size so clicks map correctly
        try:
            cv2.resizeWindow(window, display_vis.shape[1], display_vis.shape[0])
        except Exception:
            pass
        cv2.imshow(window, display_vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # next image
            current_idx = (current_idx + 1) % len(imgs)
            segmenter = FloodFillSegmenter(imgs[current_idx], os.path.join(args.input, args.output), tol=segmenter.tol, radius=segmenter.radius)
            cv2.setTrackbarPos('tol', ctrl_win, segmenter.tol)
            cv2.setTrackbarPos('radius', ctrl_win, segmenter.radius)
            cv2.setTrackbarPos('brush', ctrl_win, segmenter.brush)
            print('Loaded', imgs[current_idx])
        elif key == ord('p'):
            current_idx = (current_idx - 1) % len(imgs)
            segmenter = FloodFillSegmenter(imgs[current_idx], os.path.join(args.input, args.output), tol=segmenter.tol, radius=segmenter.radius)
            cv2.setTrackbarPos('tol', ctrl_win, segmenter.tol)
            cv2.setTrackbarPos('radius', ctrl_win, segmenter.radius)
            cv2.setTrackbarPos('brush', ctrl_win, segmenter.brush)
            print('Loaded', imgs[current_idx])
        elif key == ord('r'):
            segmenter.masks = []
            segmenter.mask_colors = []
            print('Reset masks for current image')
        elif key == ord('u'):
            if segmenter.undo():
                print('Undid last mask')
            else:
                print('No mask to undo')
        elif key == ord('m'):
            segmenter.manual_mode = not segmenter.manual_mode
            print('Manual painting mode:', 'ON' if segmenter.manual_mode else 'OFF')
        elif key == ord('c'):
            # create a new empty mask to edit manually
            empty = np.zeros((segmenter.h, segmenter.w), dtype=np.uint8)
            segmenter.masks.append(empty)
            segmenter.mask_colors.append(tuple(int(c) for c in np.random.randint(0, 256, size=3)))
            print('Created empty mask #', len(segmenter.masks))
        elif key == ord('s'):
            saved = segmenter.save_masks(save_binaries=(not args.no_binaries))
            if saved:
                print('Saved masks:', ', '.join(saved))
            else:
                print('No masks to save')
        elif key == ord('v'):
            # cycle display mode
            if display['mode'] == 'both':
                display['mode'] = 'overlay'
            elif display['mode'] == 'overlay':
                display['mode'] = 'original'
            else:
                display['mode'] = 'both'
            print('Display mode ->', display['mode'])

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
