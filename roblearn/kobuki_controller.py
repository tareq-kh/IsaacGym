def getSpeed(v, w):
    r = 0.035  # get_r()
    L = 0.23  # get_l()
    vr = (2 * v + w * L) / 2 * r
    vl = (2 * v - w * L) / 2 * r

    return vr, vl