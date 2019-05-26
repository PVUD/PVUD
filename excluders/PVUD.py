import re

import numpy as np


class Excluder:
    """
    In the CUHK dataset evaluation, we need to exclude the same PID in
    the same camera (CID).
    """
    def __init__(self, gallery_fids):
        # Setup a regexp for extracting the PID and camera (CID) form a FID.
        # Parse the gallery_set
        self.gallery_pids, self.gallery_cids = self._parse(gallery_fids)

    def __call__(self, query_fids):
        # Extract both the PIDs and CIDs from the query filenames:
        query_pids, query_cids = self._parse(query_fids)

        # Ignore same pid image within the same camera
        cid_matches = self.gallery_cids[None] == query_cids[:,None]
        pid_matches = self.gallery_pids[None] == query_pids[:,None]
        mask = np.logical_and(cid_matches, pid_matches)

        return mask

    def _parse(self, fids):
        """ Return the PIDs and CIDs extracted from the FIDs. """
        pids = []
        cids = []
        for fid in fids:
            if 'veri' in fid.lower():
                tmp = fid.split('/')[-1]
                pid = '10'+tmp.split('_')[0]
                cid = tmp.split('_')[1][1:]
            elif 'market' in fid.lower():
                tmp = fid.split('/')[-1]
                pid = '30'+tmp.split('_')[0]
                cid = tmp.split('_')[1][1]
            elif 'cuhk' in fid.lower():
                tmp = fid.split('/')[-1]
                pid = '20'+tmp.split('_')[0]
                cid = tmp.split('_')[1]
            elif 'vehicle' in fid.lower():
                tmp = fid.split('/')[-1]
                pid = '30'+tmp.split('.')[0]
                cid = '1'
            pids.append(pid)
            cids.append(cid)
        return np.asarray(pids), np.asarray(cids)
