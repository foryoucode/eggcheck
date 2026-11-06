"""Learned visual reference bank + neural defect classifier input features.
The ONLY data source is the RGB crop read from a captured frame.
No browser, webpage API or simulator metadata used.
"""
from __future__ import annotations
import numpy as np
import cv2
from .vision import prepare_gray


def encode_crop(bgr):
    im = prepare_gray(bgr)
    im = cv2.GaussianBlur(im,(3,3),.55)
    z = (im - float(im.mean())) / (float(im.std())+1e-5)
    ed = cv2.Canny(np.uint8(np.clip(im*255,0,255)),65,105).astype(np.float32)/255.
    return z.astype('float32'), ed

class NormalBank:
    def __init__(self, normal_crops=None, saved=None):
        if saved is not None:
            self.good=saved['good'].astype('float32')
            self.edges=saved['edges'].astype('float32')
            self.pca_center=saved['pca_center'].astype('float32')
            self.components=saved['components'].astype('float32')
        else:
            all_enc=[encode_crop(bgr) for bgr in normal_crops]
            self.good=np.stack([x[0] for x in all_enc]);self.edges=np.stack([x[1] for x in all_enc])
            if len(self.good)<4: raise ValueError('Need at least four good trays')
            X=self.good.reshape(len(self.good),-1)
            self.pca_center=X.mean(0).astype('float32')
            U,S,V=np.linalg.svd((X-self.pca_center).astype('float32'),full_matrices=False)
            self.components=V[:min(12,len(self.good)-2)].astype('float32')

    def as_save_dict(self):
        return {'good':self.good,'edges':self.edges,'pca_center':self.pca_center,'components':self.components}

    def get_features(self,bgr,exclude=-1):
        z,ed=encode_crop(bgr)
        # Compare visual shape with normal examples, not with any simulated defect code.
        normals=self.good if exclude<0 else np.delete(self.good,exclude,axis=0)
        normed=self.edges if exclude<0 else np.delete(self.edges,exclude,axis=0)
        d=np.abs(normals-z[None])
        mse=np.mean(d*d,axis=(1,2))
        b=d[np.argmin(mse)]
        edge_d=np.abs(normed-ed[None]);edge_best=np.mean(edge_d,axis=(1,2)).min()
        x=z.ravel();center=x-self.pca_center
        weights=self.components@center
        reconstruction=self.pca_center+weights@self.components
        recon_err=np.abs(reconstruction-x)
        scores=np.array([
            mse.min(), d.mean(axis=(1,2)).min(),
            np.percentile(b,90), np.percentile(b,99),
            np.mean(b>.7),np.mean(b>1),np.mean(b>1.5),
            edge_best,np.mean(np.abs(normed[np.argmin(np.mean(edge_d,axis=(1,2)))]-ed)>.5),
            recon_err.mean(),np.percentile(recon_err,99),np.mean(recon_err>1),
        ],dtype=np.float32)
        return scores
