import numpy as np


def channel_generator(M, K, numberr, pmax):
    # standard deviation for shadow fading
    sigma_sf_NLOS = 4  # for NLOS
    sigma_sf_LOS = 3  # for LOS
    # Prepare LOS components of the channels
    GMean = np.zeros((M, K)) + 1j * np.zeros((M, K))
    # antenna spacing in wavelength to calculate LOS components of the channels
    antennaSpacing = 1 / 2  # Half wavelength distance
    # Prepare UE positions (real part is x-axis and imaginary part is y-axis components)
    UEpositions = np.zeros((K, 1)) + 1j * np.zeros((K, 1))
    # Variable indicating how many users' locations are determined
    perBS = 0
    # UEs are located in a cell of maxDistance m x maxDistance m area
    maxDistance = 250
    # minimum distance of UEs to the BS by taking into account 8.5 height difference
    minDistance = np.sqrt(10 * 10 - 8.5 * 8.5)
    # Prepare channel gains in dB
    channelGaindB = np.zeros((K, 1))
    # continue until all the UEs' locations are determined
    while perBS < K:
        UEremaining = K - perBS
        posX = np.random.rand(UEremaining, 1) * maxDistance - maxDistance / 2
        posY = np.random.rand(UEremaining, 1) * maxDistance - maxDistance / 2
        posXY = posX + 1j * posY
        # Keep the UE if it satisfies minimum distance criterion
        posXY = posXY[np.abs(posXY) > minDistance]
        posXY = posXY.reshape(posXY.shape[0], 1)
        UEpositions[perBS:perBS + posXY.shape[0], 0:] = posXY
        # Increase the number of determined UEs
        perBS = perBS + posXY.shape[0]

    # BS height
    hBS = 10
    # UE height
    hUT = 1.5
    # effective BS and UE heights in [26]
    hBS2 = hBS - 1
    hUT2 = hUT - 1
    # breakpoint distance in [26] where carrier frequency is 2GHz
    bpdist = 4 * hBS2 * hUT2 * 20 / 3
    # 3D distances of UEs to the BS
    distancesBS = np.sqrt(np.square(np.abs(UEpositions)) + 8.5 * 8.5)

    # Prepare probabilities of LOS for UEs
    probLOSprep = np.zeros((K, 1))
    for k in np.arange(K):
        probLOSprep[k, 0] = min(18 / distancesBS[k], 1) * (1 - np.exp(-distancesBS[k] / 36)) + np.exp(
            -distancesBS[k] / 36)
    probLOS = (np.random.rand(K, 1) < probLOSprep).astype(int)
    ricianFactor = np.power(10, (5 * np.random.randn(K, 1) + 9) / 10)
    for k in np.arange(K):

        if probLOS[k, 0] == 1:
            if distancesBS[k] < bpdist:
                channelGaindB[k, 0] = -22 * np.log10(distancesBS[k]) - 28 - 20 * np.log10(2)
            else:
                channelGaindB[k, 0] = -40 * np.log10(distancesBS[k]) - 7.8 + 18 * np.log10(hBS2) + 18 * np.log10(
                    hUT2) - 2 * np.log10(2)
        else:
            channelGaindB[k, 0] = -36.7 * np.log10(distancesBS[k]) - 22.7 - 26 * np.log10(2)

    for k in np.arange(K):

        if probLOS[k, 0] == 1:
            shadowing = sigma_sf_LOS * np.random.randn(1, 1)
            channelGainShadowing = channelGaindB[k, 0] + shadowing
        else:
            shadowing = sigma_sf_NLOS * np.random.randn(1, 1)
            channelGainShadowing = channelGaindB[k, 0] + shadowing

        channelGaindB[k, 0] = channelGainShadowing

    for k in np.arange(K):
        # Angle of the UEs with respect to the BS
        angleBS = np.angle(UEpositions[k, 0])
        # Add random phase shift to the LOS components of the channels for training neural networks
        anglerandom = 2 * np.pi * np.random.rand(1)
        # normalized LOS vector by assuming uniform linear array
        GMean[:, k] = np.exp(1j * anglerandom + 1j * 2 * np.pi * np.arange(M) * np.sin(angleBS) * antennaSpacing)

        # bandwidth in Hz
    B = 20e6
    # noise figure in dB
    noiseFigure = 5
    noiseVariancedBm = -174 + 10 * np.log10(B) + noiseFigure
    # channel gain over noise in dB
    channelGainOverNoise = channelGaindB - noiseVariancedBm + 30

    # apply the heuristic uplink power control in reference [5, Section 7.1.2] with delta=20 dB
    betaMin = np.min(channelGainOverNoise[channelGainOverNoise > -np.inf])

    deltadB = 20

    differenceSNR = channelGainOverNoise - betaMin
    backoff = differenceSNR - deltadB
    backoff[backoff < 0] = 0
    # p_k in the paper
    power_coef = pmax / np.power(10, backoff / 10)

    # prepare LOS and NLOS channel gains
    channelGain_LOS = np.zeros((K, 1))
    channelGain_NLOS = np.zeros((K, 1))

    for k in np.arange(K):

        if probLOS[k, 0] == 1:  # The LoS Path exists, Rician Factor ~= 0
            channelGain_LOS[k, 0] = (ricianFactor[k, 0] / (ricianFactor[k, 0] + 1)) * np.power(10, channelGainOverNoise[
                k, 0] / 10)
            channelGain_NLOS[k, 0] = (1 / (ricianFactor[k, 0] + 1)) * np.power(10, channelGainOverNoise[k, 0] / 10)
        else:  # Pure NLoS case
            channelGain_LOS[k, 0] = 0
            channelGain_NLOS[k, 0] = np.power(10, channelGainOverNoise[k, 0] / 10)

        GMean[:, k] = np.sqrt(channelGain_LOS[k, 0]) * GMean[:, k]

    # sort the UE indices according to their channel gains for improvement in deep learning
    indexx = np.argsort(channelGainOverNoise[:, 0])
    channelGainOverNoise = channelGainOverNoise[indexx, :]
    GMean = GMean[:, indexx]
    probLOS = probLOS[indexx, :]
    channelGain_NLOS = channelGain_NLOS[indexx, :]
    power_coef = power_coef[indexx, :]

    G = np.zeros((M, K, numberr)) + 1j * np.zeros((M, K, numberr))
    for nnn in np.arange(numberr):
        W = np.sqrt(0.5) * (np.random.randn(M, K) + 1j * np.random.randn(M, K))

        G_Rayleigh = np.matmul(W, np.sqrt(np.diag(channelGain_NLOS[:, 0])))

        G[:, :, nnn] = GMean + G_Rayleigh

    return G, channelGainOverNoise, channelGain_NLOS, GMean, power_coef
