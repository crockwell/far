import cv2
import numpy as np

def recoverPose(E, points1, points2, cameraMatrix, distanceThresh, mask):
    import pdb; pdb.set_trace()
    if (points1.channels() > 1):
        npoints = points1.shape[-1]
        points1 = points1.reshape(1, npoints)
        points2 = points2.reshape(1, npoints)

    fx = cameraMatrix[0,0]
    fy = cameraMatrix[1,1]
    cx = cameraMatrix[0,2]
    cy = cameraMatrix[1,2]

    import pdb; pdb.set_trace()
    points1[:,0] = (points1[:,0] - cx) / fx
    points2[:,0] = (points2[:,0] - cx) / fx
    points1[:,1] = (points1[:,1] - cy) / fy
    points2[:,1] = (points2[:,1] - cy) / fy

    points1 = points1.T
    points2 = points2.T

    (R1, R2, t) = cv2.decomposeEssentialMat(E)
    #decomposeEssentialMat(E, R1, R2, t)
    P0, P1, P2, P3, P4 = np.zeros((3,4)), np.zeros((3,4)), np.zeros((3,4)), np.zeros((3,4)), np.zeros((3,4))
    P0[:,:3] = np.eye(3)
    P1[:,:3] = R1
    P1[:,3] =  t
    P2[:,:3] = R2
    P2[:,3] =  t
    P3[:,:3] = R1
    P3[:,3] =  -t
    P4[:,:3] = R2
    P4[:,3] =  -t

    """
    Do the chirality check.
    Notice here a threshold dist is used to filter
    out far away points (i.e. infinite points) since
    their depth may vary between positive and negative.
    """
    import pdb; pdb.set_trace()
    allTriangulations = []

    # from https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/triangulate.cpp#L346
    # which calls https://github.com/opencv/opencv/blob/4.x/modules/calib3d/src/triangulate.cpp#L54
    Q = cv2.triangulatePoints(P0, P1, points1, points2)
    #triangulatePoints(P0, P1, points1, points2, Q)
    mask1 = Q[2].mul(Q[3]) > 0
    Q[0] /= Q[3]
    Q[1] /= Q[3]
    Q[2] /= Q[3]
    Q[3] /= Q[3]
    mask1 = (Q[2] < distanceThresh) & mask1
    Q = P1 * Q
    allTriangulations.append(Q)
    mask1 = (Q[2] > 0) & mask1
    mask1 = (Q[2] < distanceThresh) & mask1

    Q = cv2.triangulatePoints(P0, P2, points1, points2)
    #triangulatePoints(P0, P1, points1, points2, Q)
    mask2 = Q[2].mul(Q[3]) > 0
    Q[0] /= Q[3]
    Q[1] /= Q[3]
    Q[2] /= Q[3]
    Q[3] /= Q[3]
    mask2 = (Q[2] < distanceThresh) & mask2
    Q = P2 * Q
    allTriangulations.append(Q)
    mask2 = (Q[2] > 0) & mask2
    mask2 = (Q[2] < distanceThresh) & mask2

    Q = cv2.triangulatePoints(P0, P3, points1, points2)
    #triangulatePoints(P0, P1, points1, points2, Q)
    mask3 = Q[2].mul(Q[3]) > 0
    Q[0] /= Q[3]
    Q[1] /= Q[3]
    Q[2] /= Q[3]
    Q[3] /= Q[3]
    mask3 = (Q[2] < distanceThresh) & mask3
    Q = P3 * Q
    allTriangulations.append(Q)
    mask3 = (Q[2] > 0) & mask3
    mask3 = (Q[2] < distanceThresh) & mask3

    Q = cv2.triangulatePoints(P0, P4, points1, points2)
    #triangulatePoints(P0, P1, points1, points2, Q)
    mask4 = Q[2].mul(Q[3]) > 0
    Q[0] /= Q[3]
    Q[1] /= Q[3]
    Q[2] /= Q[3]
    Q[3] /= Q[3]
    mask4 = (Q[2] < distanceThresh) & mask4
    Q = P4 * Q
    allTriangulations.append(Q)
    mask4 = (Q[2] > 0) & mask4
    mask4 = (Q[2] < distanceThresh) & mask4

    mask1 = mask1.t()
    mask2 = mask2.t()
    mask3 = mask3.t()
    mask4 = mask4.t()

    # If _mask is given, then use it to filter outliers.
    import pdb; pdb.set_trace()
    assert(npoints == mask.shape[1])
    mask = mask.reshape(1, npoints)
    bitwise_and(mask, mask1, mask1)
    bitwise_and(mask, mask2, mask2)
    bitwise_and(mask, mask3, mask3)
    bitwise_and(mask, mask4, mask4)

    good1 = mask1.sum()
    good2 = mask2.sum()
    good3 = mask3.sum()
    good4 = mask4.sum()

    n, R, mask, triangulatedPoints = None, None, None, None
    if good1 >= good2 and good1 >= good3 and good1 >= good4:
        triangulatedPoints = allTriangulations[0]
        R = R1
        mask = mask1
        n = good1
    elif good2 >= good1 and good2 >= good3 and good2 >= good4:
        triangulatedPoints = allTriangulations[1]
        R = R2
        mask = mask2
        n = good2
    elif good3 >= good1 and good3 >= good2 and good3 >= good4:
        triangulatedPoints = allTriangulations[2]
        t = -t
        R = R1
        mask = mask3
        n = good3
    else:
        triangulatedPoints = allTriangulations[3]
        t = -t
        R = R2
        mask = mask4
        n = good4
    
    return n, R, t, mask


"""
cv2 implementation

int cv::recoverPose( InputArray E, InputArray _points1, InputArray _points2,
                            InputArray _cameraMatrix, OutputArray _R, OutputArray _t, double distanceThresh,
                     InputOutputArray _mask, OutputArray triangulatedPoints)
{
    CV_INSTRUMENT_REGION();

    Mat points1, points2, cameraMatrix;
    _points1.getMat().convertTo(points1, CV_64F);
    _points2.getMat().convertTo(points2, CV_64F);
    _cameraMatrix.getMat().convertTo(cameraMatrix, CV_64F);

    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
                              points1.type() == points2.type());

    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3 && cameraMatrix.channels() == 1);

    if (points1.channels() > 1)
    {
        points1 = points1.reshape(1, npoints);
        points2 = points2.reshape(1, npoints);
    }

    double fx = cameraMatrix.at<double>(0,0);
    double fy = cameraMatrix.at<double>(1,1);
    double cx = cameraMatrix.at<double>(0,2);
    double cy = cameraMatrix.at<double>(1,2);

    points1.col(0) = (points1.col(0) - cx) / fx;
    points2.col(0) = (points2.col(0) - cx) / fx;
    points1.col(1) = (points1.col(1) - cy) / fy;
    points2.col(1) = (points2.col(1) - cy) / fy;

    points1 = points1.t();
    points2 = points2.t();

    Mat R1, R2, t;
    decomposeEssentialMat(E, R1, R2, t);
    Mat P0 = Mat::eye(3, 4, R1.type());
    Mat P1(3, 4, R1.type()), P2(3, 4, R1.type()), P3(3, 4, R1.type()), P4(3, 4, R1.type());
    P1(Range::all(), Range(0, 3)) = R1 * 1.0; P1.col(3) = t * 1.0;
    P2(Range::all(), Range(0, 3)) = R2 * 1.0; P2.col(3) = t * 1.0;
    P3(Range::all(), Range(0, 3)) = R1 * 1.0; P3.col(3) = -t * 1.0;
    P4(Range::all(), Range(0, 3)) = R2 * 1.0; P4.col(3) = -t * 1.0;

    // Do the chirality check.
    // Notice here a threshold dist is used to filter
    // out far away points (i.e. infinite points) since
    // their depth may vary between positive and negative.
    std::vector<Mat> allTriangulations(4);
    Mat Q;

    triangulatePoints(P0, P1, points1, points2, Q);
    if(triangulatedPoints.needed())
        Q.copyTo(allTriangulations[0]);
    Mat mask1 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask1 = (Q.row(2) < distanceThresh) & mask1;
    Q = P1 * Q;
    mask1 = (Q.row(2) > 0) & mask1;
    mask1 = (Q.row(2) < distanceThresh) & mask1;

    triangulatePoints(P0, P2, points1, points2, Q);
    if(triangulatedPoints.needed())
        Q.copyTo(allTriangulations[1]);
    Mat mask2 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask2 = (Q.row(2) < distanceThresh) & mask2;
    Q = P2 * Q;
    mask2 = (Q.row(2) > 0) & mask2;
    mask2 = (Q.row(2) < distanceThresh) & mask2;

    triangulatePoints(P0, P3, points1, points2, Q);
    if(triangulatedPoints.needed())
        Q.copyTo(allTriangulations[2]);
    Mat mask3 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask3 = (Q.row(2) < distanceThresh) & mask3;
    Q = P3 * Q;
    mask3 = (Q.row(2) > 0) & mask3;
    mask3 = (Q.row(2) < distanceThresh) & mask3;

    triangulatePoints(P0, P4, points1, points2, Q);
    if(triangulatedPoints.needed())
        Q.copyTo(allTriangulations[3]);
    Mat mask4 = Q.row(2).mul(Q.row(3)) > 0;
    Q.row(0) /= Q.row(3);
    Q.row(1) /= Q.row(3);
    Q.row(2) /= Q.row(3);
    Q.row(3) /= Q.row(3);
    mask4 = (Q.row(2) < distanceThresh) & mask4;
    Q = P4 * Q;
    mask4 = (Q.row(2) > 0) & mask4;
    mask4 = (Q.row(2) < distanceThresh) & mask4;

    mask1 = mask1.t();
    mask2 = mask2.t();
    mask3 = mask3.t();
    mask4 = mask4.t();

    // If _mask is given, then use it to filter outliers.
    if (!_mask.empty())
    {
        Mat mask = _mask.getMat();
        CV_Assert(npoints == mask.checkVector(1));
        mask = mask.reshape(1, npoints);
        bitwise_and(mask, mask1, mask1);
        bitwise_and(mask, mask2, mask2);
        bitwise_and(mask, mask3, mask3);
        bitwise_and(mask, mask4, mask4);
    }
    if (_mask.empty() && _mask.needed())
    {
        _mask.create(mask1.size(), CV_8U);
    }

    CV_Assert(_R.needed() && _t.needed());
    _R.create(3, 3, R1.type());
    _t.create(3, 1, t.type());

    int good1 = countNonZero(mask1);
    int good2 = countNonZero(mask2);
    int good3 = countNonZero(mask3);
    int good4 = countNonZero(mask4);

    if (good1 >= good2 && good1 >= good3 && good1 >= good4)
    {
        if(triangulatedPoints.needed()) allTriangulations[0].copyTo(triangulatedPoints);
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask1.copyTo(_mask);
        return good1;
    }
    else if (good2 >= good1 && good2 >= good3 && good2 >= good4)
    {
        if(triangulatedPoints.needed()) allTriangulations[1].copyTo(triangulatedPoints);
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask2.copyTo(_mask);
        return good2;
    }
    else if (good3 >= good1 && good3 >= good2 && good3 >= good4)
    {
        if(triangulatedPoints.needed()) allTriangulations[2].copyTo(triangulatedPoints);
        t = -t;
        R1.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask3.copyTo(_mask);
        return good3;
    }
    else
    {
        if(triangulatedPoints.needed()) allTriangulations[3].copyTo(triangulatedPoints);
        t = -t;
        R2.copyTo(_R);
        t.copyTo(_t);
        if (_mask.needed()) mask4.copyTo(_mask);
        return good4;
    }
}
"""