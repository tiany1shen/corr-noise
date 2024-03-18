import numpy as np 
from scipy import linalg

def calc_fid(features_gen, features_gt):
    """ input: feature list of vectors
    """

    # print(kps_gen.shape)
    # print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]
    kps_gen = features_gen
    kps_gt = features_gt

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt
    
    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'


    # print(mu1, mu2)
    diff = mu1 - mu2
    
    eps = 1e-6
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)