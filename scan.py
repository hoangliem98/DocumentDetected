from pyimagesearch import transform
from pyimagesearch import imutils
from scipy.spatial import distance as dist
from matplotlib.patches import Polygon
import polygon_interacter as poly_i
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import imutils
import cv2 as cv

import argparse
import os

class DocScanner(object):
    def __init__(self, interactive=False, MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
        """
        Args:
            interactive (boolean): Nếu đúng người dùng có thể điều chỉnh các đường trong màn hình pyplot.
            MIN_QUAD_AREA_RATIO (float): Contour sẽ bị từ chối nếu các góc của nó không tạo thành 1 tứ giác. Defaults to 0.25.
            MAX_QUAD_ANGLE_RANGE (int):  Contour cũng sẽ bị từ chối nếu các góc vượt quá MAX_QUAD_ANGLE_RANGE. Defaults to 40.
        """        
        self.interactive = interactive
        self.MIN_QUAD_AREA_RATIO = MIN_QUAD_AREA_RATIO
        self.MAX_QUAD_ANGLE_RANGE = MAX_QUAD_ANGLE_RANGE        

    # Lọc các góc
    def filter_corners(self, corners, min_dist=20):
        def predicate(representatives, corner):
            return all(dist.euclidean(representative, corner) >= min_dist
                       for representative in representatives)

        filtered_corners = []
        for c in corners:
            if predicate(filtered_corners, c):
                filtered_corners.append(c)
        return filtered_corners
    #Góc giữa 2 cạnh
    def angle_between_vectors_degrees(self, u, v):
        return np.degrees(
            math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(self, p1, p2, p3):
        """Trả về góc giữa đoạn p2 đến p1 và đoạn p2 đến p3"""
        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return self.angle_between_vectors_degrees(avec, cvec)

    def angle_range(self, quad):
        tl, tr, br, bl = quad
        ura = self.get_angle(tl[0], tr[0], br[0])
        ula = self.get_angle(bl[0], tl[0], tr[0])
        lra = self.get_angle(tr[0], br[0], bl[0])
        lla = self.get_angle(br[0], bl[0], tl[0])

        angles = [ura, ula, lra, lla]
        return np.ptp(angles)          

    def get_corners(self, img):
        """
        Trả về 1 danh sách các góc được tìm thấy từ ảnh đầu vào. Với hình ảnh đã được xử lý và lọc, 
        có thể cho ra 10 góc có khả năng nhất.
        """
        lsd = cv.ximgproc.createFastLineDetector()
        lines = lsd.detect(img)

        corners = []
        if lines is not None:
            # separate out the horizontal and vertical lines, and draw them back onto separate canvases
            lines = lines.squeeze().astype(np.int32).tolist()
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for line in lines:
                x1, y1, x2, y2 = line
                if abs(x2 - x1) > abs(y2 - y1):
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0])
                    cv.line(horizontal_lines_canvas, (max(x1 - 5, 0), y1), (min(x2 + 5, img.shape[1] - 1), y2), 255, 2)
                else:
                    (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1])
                    cv.line(vertical_lines_canvas, (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, img.shape[0] - 1)), 255, 2)

            lines = []

            # lấy chiều ngang (connected-components -> bounding boxes -> final lines)
            contours = cv.findContours(horizontal_lines_canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours = contours[0]
            contours = sorted(contours, key=lambda c: cv.arcLength(c, True), reverse=True)[:2]
            horizontal_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_x = np.amin(contour[:, 0], axis=0) + 2
                max_x = np.amax(contour[:, 0], axis=0) - 2
                left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
                right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))
                lines.append((min_x, left_y, max_x, right_y))
                cv.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)
                corners.append((min_x, left_y))
                corners.append((max_x, right_y))

            # lấy chiều dọc (connected-components -> bounding boxes -> final lines)
            contours = cv.findContours(vertical_lines_canvas, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            contours = contours[0]
            contours = sorted(contours, key=lambda c: cv.arcLength(c, True), reverse=True)[:2]
            vertical_lines_canvas = np.zeros(img.shape, dtype=np.uint8)
            for contour in contours:
                contour = contour.reshape((contour.shape[0], contour.shape[2]))
                min_y = np.amin(contour[:, 1], axis=0) + 2
                max_y = np.amax(contour[:, 1], axis=0) - 2
                top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
                bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))
                lines.append((top_x, min_y, bottom_x, max_y))
                cv.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)
                corners.append((top_x, min_y))
                corners.append((bottom_x, max_y))

            # tìm các góc
            corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
            corners += zip(corners_x, corners_y)

        # lọc các góc
        corners = self.filter_corners(corners)
        return corners

    def is_valid_contour(self, cnt, IM_WIDTH, IM_HEIGHT):
        return (len(cnt) == 4 and cv.contourArea(cnt) > IM_WIDTH * IM_HEIGHT * self.MIN_QUAD_AREA_RATIO 
            and self.angle_range(cnt) < self.MAX_QUAD_ANGLE_RANGE)

    # Lấy khung tự động phần doc  
    def get_contour(self, rescaled_image):
        # khai báo các thông số cần dùng
        MORPH = 9
        CANNY = 84
        HOUGH = 25

        IM_HEIGHT, IM_WIDTH, _ = rescaled_image.shape

        gray = cv.cvtColor(rescaled_image, cv.COLOR_BGR2GRAY) #Chuyển ảnh sang ảnh gray
        gray = cv.GaussianBlur(gray, (7,7), 0) #Làm mờ để khử nhiễu
        kernel = cv.getStructuringElement(cv.MORPH_RECT,(MORPH,MORPH)) #Khai báo kernel
        dilated = cv.dilate(gray, kernel) #Làm đậm nét

        # Tìm cạnh trong ảnh sử dụng Canny
        edged = cv.Canny(dilated, 0, CANNY)
        test_corners = self.get_corners(edged)

        approx_contours = []

        if len(test_corners) >= 4:
            quads = []

            for quad in itertools.combinations(test_corners, 4):
                points = np.array(quad)
                points = transform.order_points(points)
                points = np.array([[p] for p in points], dtype = "int32")
                quads.append(points)
            
            # lấy 5 khung tứ giác tốt nhất 
            quads = sorted(quads, key=cv.contourArea, reverse=True)[:5]
            # sắp xếp các điểm của tứ giác đúng thứ tự, loại bỏ các điểm thừa
            quads = sorted(quads, key=self.angle_range)

            approx = quads[0]
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)

        # lấy kết quả tốt hơn
        (cnts, hierarchy) = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)[:5]

        for c in cnts:
            # lấy khung doc gần đúng
            approx = cv.approxPolyDP(c, 80, True)
            if self.is_valid_contour(approx, IM_WIDTH, IM_HEIGHT):
                approx_contours.append(approx)
                break

        # Không tìm được viền, lấy toàn bộ ảnh
        if not approx_contours:
            TOP_RIGHT = (IM_WIDTH, 0)
            BOTTOM_RIGHT = (IM_WIDTH, IM_HEIGHT)
            BOTTOM_LEFT = (0, IM_HEIGHT)
            TOP_LEFT = (0, 0)
            screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])

        else:
            screenCnt = max(approx_contours, key=cv.contourArea)
            
        return screenCnt.reshape(4, 2)

    def interactive_get_contour(self, screenCnt, rescaled_image):
        poly = Polygon(screenCnt, animated=True, fill=False, color="yellow", linewidth=5)
        fig, ax = plt.subplots(num = "Cắt ảnh")
        ax.add_patch(poly)
        ax.set_title('Kéo các điểm để tùy chỉnh. \nĐóng cửa sổ để kết thúc.')
        p = poly_i.PolygonInteractor(ax, poly)
        plt.imshow(rescaled_image)
        plt.show()

        new_points = p.get_poly_points()[:4]
        new_points = np.array([[p] for p in new_points], dtype = "int32")
        return new_points.reshape(4, 2)

    def resized(self, img):
        width = int(img.shape[1] * 0.2)
        height = int(img.shape[0] * 0.2)
        resized = cv.resize(img, (width, height), interpolation = cv.INTER_AREA)
        return resized

    def scan(self, image_path):

        RESCALED_HEIGHT = 500.0
        OUTPUT_DIR = 'output'

        #load ảnh
        image = cv.imread(image_path)

        assert(image is not None)

        #tính toán tỉ lệ & resize
        ratio = image.shape[0] / RESCALED_HEIGHT
        orig = image.copy()
        rescaled_image = imutils.resize(image, height = int(RESCALED_HEIGHT))
        
        # lấy viền phần doc
        screenCnt = self.get_contour(rescaled_image)

        if self.interactive:
            screenCnt = self.interactive_get_contour(screenCnt, rescaled_image)

        # Lấy phần khung xác định cuối cùng
        warped = transform.four_point_transform(orig, screenCnt * ratio)
    
        # Xoay ảnh
        win_name = "Press 'X' to rotate image. Press 'N' to continue"
        true = True
        warpedTmp = warped.copy()
        while true:
            cv.imshow(win_name, self.resized(warpedTmp))
            k = cv.waitKey(0)
            if k in [ord('x')]:
                for angel in np.arange(90, 360, 90):
                    rotated = imutils.rotate_bound(warped, angel)
                    warped = rotated
                    resized = self.resized(rotated)
                    cv.imshow(win_name, resized)
                    k2 = cv.waitKey(0)  
                    if k2 == ord('n'):	
                        cv.destroyWindow(win_name)
                        true = False
                        break
            elif k == ord('n'):
                cv.destroyWindow(win_name)  
                true = False
                break

        # Phần filter lại ảnh
        # Chuyến thành ảnh gray
        gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

        # làm mịn ảnh
        sharpen = cv.GaussianBlur(gray, (0,0), 3)
        sharpen = cv.addWeighted(gray, 1.5, sharpen, -0.5, 0)

        # lọc ảnh đen trắng
        thresh = cv.adaptiveThreshold(sharpen, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, 15)

        # lưu ảnh đã fill
        basename = os.path.basename(image_path)
        cv.imwrite(image_path, thresh)
        print("Đã lưu " + basename)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to single image to be scanned")
    ap.add_argument("-i", action='store_true',
        help = "Flag for manually verifying and/or setting document corners")

    args = vars(ap.parse_args())
    im_file_path = args["image"]
    interactive_mode = args["i"]

    scanner = DocScanner(interactive_mode)
    scanner.scan(im_file_path)
    #DocScanner().scan('notepad.jpg')
