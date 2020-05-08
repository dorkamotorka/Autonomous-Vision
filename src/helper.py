
def HomogenousCoord(pts):
    # [x,y] -> [x,y,1]
    pts = np.append(pts, np.ones([pts.shape[0], 1]), axis=1)# commented numpy array in feats_extract
    return pts
