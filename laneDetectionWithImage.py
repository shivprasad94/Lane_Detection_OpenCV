import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from skimage import morphology
from collections import deque

class LaneDetectorOnImage:
    #To
    def display(self, img, title, color=1):
        if color:
            plt.imshow(img)
        else:
            plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def camera_calibration(self, folder, nx, ny, choice):
        objpoints = []  # 3D
        imgpoints = []
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        assert len(folder) != 0
        for fname in folder:
            img = cv2.imread(fname)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
            img_sz = gray.shape[::-1]
            if ret:
                imgpoints.append(corners)
                objpoints.append(objp)
                if choice:
                    draw_corners = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    self.display(draw_corners, 'Found all corners:{}'.format(ret))
        if len(objpoints) == len(imgpoints) and len(objpoints) != 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_sz, None, None)
            return {'ret': ret, 'cameraMatrix': mtx, 'distorsionCoeff': dist,
                    'rotationVec': rvecs, 'translationVec': tvecs}
        else:
            raise cv2.Error('Camera Calibration failed')

    def correction(self, image, calib_params, showMe=0):
        corrected = cv2.undistort(image, calib_params['cameraMatrix'], calib_params['distorsionCoeff'],
                                  None, calib_params['cameraMatrix'])
        if showMe:
            self.display(image, 'Original', color=1)
            self.display(corrected, 'After correction', color=1)
        return corrected

    def directional_gradient(self, img, direction='x', thresh=[0, 255]):
        if direction == 'x':
            sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        elif direction == 'y':
            sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        sobel_abs = np.absolute(sobel)  # absolute value
        scaled_sobel = np.uint8(sobel_abs * 255 / np.max(sobel_abs))
        binary_output = np.zeros_like(sobel)
        binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        return binary_output

    def color_binary(self, img, dst_format='HLS', ch=2, ch_thresh=[0, 255]):
        if dst_format == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            ch_binary = np.zeros_like(img[:, :, int(ch - 1)])
            ch_binary[(img[:, :, int(ch - 1)] >= ch_thresh[0]) & (img[:, :, int(ch - 1)] <= ch_thresh[1])] = 1
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            ch_binary = np.zeros_like(img[:, :, int(ch - 1)])
            ch_binary[(img[:, :, int(ch - 1)] >= ch_thresh[0]) & (img[:, :, int(ch - 1)] <= ch_thresh[1])] = 1
        return ch_binary

    def birdView(self, img, M):
        img_sz = (img.shape[1], img.shape[0])
        img_warped = cv2.warpPerspective(img, M, img_sz, flags=cv2.INTER_LINEAR)
        return img_warped

    def perspective_transform(self, src_pts, dst_pts):
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        return {'M': M, 'Minv': Minv}

    def find_centroid(self, image, peak_thresh, window, choice):
        # crop image to window dimension
        mask_window = image[round(window['y0'] - window['height']):round(window['y0']),
                      round(window['x0']):round(window['x0'] + window['width'])]
        histogram = np.sum(mask_window, axis=0)
        centroid = np.argmax(histogram)
        hotpixels_cnt = np.sum(histogram)
        peak_intensity = histogram[centroid]
        if peak_intensity <= peak_thresh:
            centroid = int(round(window['x0'] + window['width'] / 2))
            peak_intensity = 0
        else:
            centroid = int(round(centroid + window['x0']))
        if choice:
            plt.plot(histogram)
            plt.title('Histogram')
            plt.xlabel('horzontal position')
            plt.ylabel('hot pixels count')
            plt.show()
        return (centroid, peak_intensity, hotpixels_cnt)

    def find_starter_centroids(self, image, choice, x0, peak_thresh):
        window = {'x0': x0, 'y0': image.shape[0], 'width': image.shape[1] / 2, 'height': image.shape[0] / 2}
        # get centroid
        centroid, peak_intensity, _ = self.find_centroid(image, peak_thresh, window, choice)
        if peak_intensity < peak_thresh:
            window['height'] = image.shape[0]
            centroid, peak_intensity, _ = self.find_centroid(image, peak_thresh, window, choice)
        return {'centroid': centroid, 'intensity': peak_intensity}

    def run_sliding_window(self, image, centroid_starter, sliding_window_specs, peak_thresh, choice):
        # Initialize sliding window
        window = {'x0': centroid_starter - int(sliding_window_specs['width'] / 2),
                  'y0': image.shape[0], 'width': sliding_window_specs['width'],
                  'height': round(image.shape[0] / sliding_window_specs['n_steps'])}
        hotpixels_log = {'x': [], 'y': []}
        centroids_log = []
        if choice:
            out_img = (np.dstack((image, image, image)) * 255).astype('uint8')
        for step in range(sliding_window_specs['n_steps']):
            if window['x0'] < 0: window['x0'] = 0
            if (window['x0'] + sliding_window_specs['width']) > image.shape[1]:
                window['x0'] = image.shape[1] - sliding_window_specs['width']
            centroid, peak_intensity, hotpixels_cnt = self.find_centroid(image, peak_thresh, window, choice)
            if step == 0:
                starter_centroid = centroid
            if hotpixels_cnt / (window['width'] * window['height']) > 0.6:
                window['width'] = window['width'] * 2
                window['x0'] = round(window['x0'] - window['width'] / 2)
                if (window['x0'] < 0): window['x0'] = 0
                if (window['x0'] + window['width']) > image.shape[1]:
                    window['x0'] = image.shape[1] - window['width']
                centroid, peak_intensity, hotpixels_cnt = self.find_centroid(image, peak_thresh, window, choice)


            mask_window = np.zeros_like(image)
            mask_window[window['y0'] - window['height']:window['y0'],
            window['x0']:window['x0'] + window['width']] \
                = image[window['y0'] - window['height']:window['y0'],
                  window['x0']:window['x0'] + window['width']]

            hotpixels = np.nonzero(mask_window)

            hotpixels_log['x'].extend(hotpixels[0].tolist())
            hotpixels_log['y'].extend(hotpixels[1].tolist())
            # update record of centroid
            centroids_log.append(centroid)

            if choice:
                cv2.rectangle(out_img,
                              (window['x0'], window['y0'] - window['height']),
                              (window['x0'] + window['width'], window['y0']), (0, 255, 0), 2)

                if int(window['y0']) == 68:
                    plt.imshow(out_img)
                    plt.show()
            # set next position of window and use standard sliding window width
            window['width'] = sliding_window_specs['width']
            window['x0'] = round(centroid - window['width'] / 2)
            window['y0'] = window['y0'] - window['height']
        return hotpixels_log

    def MahalanobisDist(self, x, y):
        covariance_xy = np.cov(x, y, rowvar=0)
        inv_covariance_xy = np.linalg.inv(covariance_xy)
        xy_mean = np.mean(x), np.mean(y)
        x_diff = np.array([x_i - xy_mean[0] for x_i in x])
        y_diff = np.array([y_i - xy_mean[1] for y_i in y])
        diff_xy = np.transpose([x_diff, y_diff])

        md = []
        for i in range(len(diff_xy)):
            md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]), inv_covariance_xy), diff_xy[i])))
        return md

    def MD_removeOutliers(self, x, y, MD_thresh):
        MD = self.MahalanobisDist(x, y)
        threshold = np.mean(MD) * MD_thresh
        nx, ny, outliers = [], [], []
        for i in range(len(MD)):
            if MD[i] <= threshold:
                nx.append(x[i])
                ny.append(y[i])
            else:
                outliers.append(i)
        return (nx, ny)

    def update_tracker(self, bestfit, bestfit_real, radOfCurv_tracker, allx, ally, tracker, new_value):
        if tracker == 'bestfit':
            bestfit['a0'].append(new_value['a0'])
            bestfit['a1'].append(new_value['a1'])
            bestfit['a2'].append(new_value['a2'])
        elif tracker == 'bestfit_real':
            bestfit_real['a0'].append(new_value['a0'])
            bestfit_real['a1'].append(new_value['a1'])
            bestfit_real['a2'].append(new_value['a2'])
        elif tracker == 'radOfCurvature':
            radOfCurv_tracker.append(new_value)
        elif tracker == 'hotpixels':
            allx.append(new_value['x'])
            ally.append(new_value['y'])

    # fit to polynomial in pixel space
    def polynomial_fit(self, data):
        a2, a1, a0 = np.polyfit(data['x'], data['y'], 2)
        return {'a0': a0, 'a1': a1, 'a2': a2}

    def predict_line(self, x0, xmax, coeffs):
        x_pts = np.linspace(x0, xmax - 1, num=xmax)
        pred = coeffs['a2'] * x_pts ** 2 + coeffs['a1'] * x_pts + coeffs['a0']
        return np.column_stack((x_pts, pred))

    def compute_radOfCurvature(self, coeffs, pt):
        return ((1 + (2 * coeffs['a2'] * pt + coeffs['a1']) ** 2) ** 1.5) / np.absolute(2 * coeffs['a2'])

    def executeProcessing(self, choice):
        nx = 9
        ny = 6
        folder_calibration = glob.glob("C:/2_My_Project/17_Lane_detector/InputImage/view/view[1-3].jpg")
        calib_params = self.camera_calibration(folder_calibration, nx, ny, choice)
        print('RMS Error of Camera View:{:.3f}'.format(calib_params['ret']))
        print('This number must be between 0.1 and 1.0')
        # imgs_tests = glob.glob("Images/*.jpg")
        # original_img = np.random.choice(imgs_tests)
        original_img = cv2.imread("C:/2_My_Project/17_Lane_detector/InputImage/Images/straight_lines1.jpg")
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        corr_img = self.correction(original_img, calib_params, showMe=1)

        gray_ex = cv2.cvtColor(corr_img, cv2.COLOR_RGB2GRAY)
        if choice:
            self.display(gray_ex, 'Apply Camera Correction', color=0)

        gradx_thresh = [25, 255]
        gradx = self.directional_gradient(gray_ex, direction='x', thresh=gradx_thresh)
        if choice:
            self.display(gradx, 'Gradient x', color=0)

        ch_thresh = [50, 255]
        ch3_hls_binary = self.color_binary(corr_img, dst_format='HLS', ch=3, ch_thresh=ch_thresh)
        if choice:
            self.display(ch3_hls_binary, 'HLS to Binary S', color=0)

        combined_output = np.zeros_like(gradx)
        combined_output[((gradx == 1) | (ch3_hls_binary == 1))] = 1
        if choice:
            self.display(combined_output, 'Combined output', color=0)

        mask = np.zeros_like(combined_output)
        vertices = np.array([[(100, 720), (545, 470), (755, 470), (1290, 720)]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 1)
        masked_image = cv2.bitwise_and(combined_output, mask)
        if choice:
            self.display(masked_image, 'Masked', color=0)

        min_sz = 50
        cleaned = morphology.remove_small_objects(masked_image.astype('bool'), min_size=min_sz, connectivity=2)
        if choice:
            self.display(cleaned, 'cleaned', color=0)

        # original image to bird view (transformation)
        src_pts = np.float32([[240, 720], [575, 470], [735, 470], [1200, 720]])
        dst_pts = np.float32([[240, 720], [240, 0], [1200, 0], [1200, 720]])
        transform_matrix = self.perspective_transform(src_pts, dst_pts)
        warped_image = self.birdView(cleaned * 1.0, transform_matrix['M'])
        self.display(cleaned, 'undistorted', color=0)
        if choice:
            self.display(warped_image, 'BirdViews', color=0)

        bottom_crop = -40
        warped_image = warped_image[0:bottom_crop, :]

        # if number of histogram pixels in window is below 10,condisder them as noise and does not attempt to get centroid
        peak_thresh = 10

        centroid_starter_right = self.find_starter_centroids(warped_image, choice, x0=warped_image.shape[1] / 2,
                                                             peak_thresh=peak_thresh)
        centroid_starter_left = self.find_starter_centroids(warped_image, choice, x0=0, peak_thresh=peak_thresh)
        peak_thresh = 10
        sliding_window_specs = {'width': 120, 'n_steps': 10}
        log_lineLeft = self.run_sliding_window(warped_image, centroid_starter_left['centroid'], sliding_window_specs,
                                               peak_thresh, choice)
        log_lineRight = self.run_sliding_window(warped_image, centroid_starter_right['centroid'], sliding_window_specs,
                                                peak_thresh, choice)

        MD_thresh = 1.8
        log_lineLeft['x'], log_lineLeft['y'] = \
            self.MD_removeOutliers(log_lineLeft['x'], log_lineLeft['y'], MD_thresh)
        log_lineRight['x'], log_lineRight['y'] = \
            self.MD_removeOutliers(log_lineRight['x'], log_lineRight['y'], MD_thresh)

        buffer_sz = 5
        allx = deque([], maxlen=buffer_sz)
        ally = deque([], maxlen=buffer_sz)
        bestfit = {'a0': deque([], maxlen=buffer_sz),
                   'a1': deque([], maxlen=buffer_sz),
                   'a2': deque([], maxlen=buffer_sz)}
        bestfit_real = {'a0': deque([], maxlen=buffer_sz),
                        'a1': deque([], maxlen=buffer_sz),
                        'a2': deque([], maxlen=buffer_sz)}
        radOfCurv_tracker = deque([], maxlen=buffer_sz)

        self.update_tracker(bestfit, bestfit_real, radOfCurv_tracker, allx, ally, 'hotpixels', log_lineRight)
        self.update_tracker(bestfit, bestfit_real, radOfCurv_tracker, allx, ally, 'hotpixels', log_lineLeft)
        multiframe_r = {'x': [val for sublist in allx for val in sublist],
                        'y': [val for sublist in ally for val in sublist]}
        multiframe_l = {'x': [val for sublist in allx for val in sublist],
                        'y': [val for sublist in ally for val in sublist]}

        # merters per pixel in y or x dimension
        ym_per_pix = 12 / 450
        xm_per_pix = 3.7 / 911
        fit_lineLeft = self.polynomial_fit(multiframe_l)
        fit_lineLeft_real = self.polynomial_fit({'x': [i * ym_per_pix for i in multiframe_l['x']],
                                                 'y': [i * xm_per_pix for i in multiframe_l['y']]})
        fit_lineRight = self.polynomial_fit(multiframe_r)
        fit_lineRight_real = self.polynomial_fit({'x': [i * ym_per_pix for i in multiframe_r['x']],
                                                  'y': [i * xm_per_pix for i in multiframe_r['y']]})

        fit_lineRight_singleframe = self.polynomial_fit(log_lineRight)
        fit_lineLeft_singleframe = self.polynomial_fit(log_lineLeft)
        var_pts = np.linspace(0, corr_img.shape[0] - 1, num=corr_img.shape[0])
        pred_lineLeft_singleframe = self.predict_line(0, corr_img.shape[0], fit_lineLeft_singleframe)
        pred_lineRight_sigleframe = self.predict_line(0, corr_img.shape[0], fit_lineRight_singleframe)

        if choice:
            plt.plot(pred_lineLeft_singleframe[:, 1], pred_lineLeft_singleframe[:, 0], 'b-', label='singleframe',
                     linewidth=1)
            plt.plot(pred_lineRight_sigleframe[:, 1], pred_lineRight_sigleframe[:, 0], 'b-', label='singleframe',
                     linewidth=1)
            plt.show()

        pt_curvature = corr_img.shape[0]
        radOfCurv_r = self.compute_radOfCurvature(fit_lineRight_real, pt_curvature * ym_per_pix)
        radOfCurv_l = self.compute_radOfCurvature(fit_lineLeft_real, pt_curvature * ym_per_pix)
        average_radCurv = (radOfCurv_r + radOfCurv_l) / 2

        center_of_lane = (pred_lineLeft_singleframe[:, 1][-1] + pred_lineRight_sigleframe[:, 1][-1]) / 2
        offset = (corr_img.shape[1] / 2 - center_of_lane) * xm_per_pix

        side_pos = 'right'
        if offset < 0:
            side_pos = 'left'
        wrap_zero = np.zeros_like(gray_ex).astype(np.uint8)
        color_wrap = np.dstack((wrap_zero, wrap_zero, wrap_zero))
        left_fitx = fit_lineLeft_singleframe['a2'] * var_pts ** 2 + fit_lineLeft_singleframe['a1'] * var_pts + \
                    fit_lineLeft_singleframe['a0']
        right_fitx = fit_lineRight_singleframe['a2'] * var_pts ** 2 + fit_lineRight_singleframe['a1'] * var_pts + \
                     fit_lineRight_singleframe['a0']
        pts_left = np.array([np.transpose(np.vstack([left_fitx, var_pts]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, var_pts])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_wrap, np.int_([pts]), (0, 255, 0))
        cv2.putText(color_wrap, '|', (int(corr_img.shape[1] / 2), corr_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 0, 255), 8)
        cv2.putText(color_wrap, '|', (int(center_of_lane), corr_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 8)
        newwrap = cv2.warpPerspective(color_wrap, transform_matrix['Minv'], (corr_img.shape[1], corr_img.shape[0]))
        result = cv2.addWeighted(corr_img, 1, newwrap, 0.3, 0)
        cv2.putText(result, 'Vehicle is' + str(round(offset, 3)) + 'm' + side_pos + 'of center',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
        cv2.putText(result, 'Radius of curvature:' + str(round(average_radCurv)) + 'm', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
        # if choice:
        plt.title('Final Result')
        plt.imshow(result)
        plt.axis('off')
        plt.show()


