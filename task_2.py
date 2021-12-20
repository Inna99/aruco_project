import numpy as np
import cv2


if __name__ == '__main__':
    boardWidth = 7
    boardHeight = 7
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((boardHeight * boardWidth, 3), np.float32)
    objp[:, :2] = np.mgrid[0:boardWidth, 0:boardHeight].T.reshape(-1, 2)

    camera_matrix = np.load('calibration_matrix.npy')
    dist_coefs = np.load('distortion_coefficients.npy')
    print('calibration_matri')
    print(camera_matrix)
    print('distortion_coefficients')
    print(dist_coefs)

    dirpath = './Images/test_img/'
    square_size = 0.033

    fname = './Images/test_img/photo_2021-12-18_14-49-15.jpg'

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (boardWidth, boardHeight), None)
    objp = objp.astype('float32')
    corners = corners.astype('float32')
    _, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coefs)

    print("\nR_vec_test:")
    print(rvec)

    print("\nT_vec_test:")
    print(tvec)

    # Calculate euler angle
    print("\nR_matrix:")
    R_matrix, _ = cv2.Rodrigues(rvec)
    print(R_matrix)

    print("\nCamera position")
    cameraPosition = -np.matrix(R_matrix).T * np.matrix(tvec)
    print(cameraPosition)

    cv2.imshow('Test image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
