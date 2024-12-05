import numpy as np


class SplitZones:
    """
    A bounding volume data structure that splits a mesh into sub-regions to quickly find triangles near a point.
    If a mesh has a large number of triangles (e.g. 10000), and we need to search for triangles at a certain location
    within the mesh, this data structure allows us to find a much-reduced subset of the triangles (~1% full number of
    triangles) near a certain point.

    :param tri_obj: TriBase mesh object
    :param splits: The number of sub-regions along each axis. For example, split=8 will split the mesh into an 8x8x8
                   grid for a total of 512 sub-regions.
    :param overlap_factor: Expand each sub-region's dimensions by this percentage, to prevent edge cases where a
                           triangle bordering one sub-region is not found in a search for a point in an adjacent
                           sub-region.
    """
    def __init__(self, tri_obj, splits=10, overlap_factor=0.2):
        tri_obj.GetTriNodes()
        self.splits = splits
        self.min_corner = np.array((np.min(tri_obj.TriX), np.min(tri_obj.TriY), np.min(tri_obj.TriZ)))
        self.max_corner = np.array((np.max(tri_obj.TriX), np.max(tri_obj.TriY), np.max(tri_obj.TriZ)))
        self.bb_size = self.max_corner - self.min_corner
        # Nx3x3 tensor of triangle coordinates [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]
        coords = np.stack((tri_obj.TriX, tri_obj.TriY, tri_obj.TriZ), axis=2)
        # find AABB of each triangle
        tri_boxes = np.stack((np.min(coords, axis=1), np.max(coords, axis=1)), axis=1)
        # into box-space (all coordinates between 0 and 1)
        coords_box = (tri_boxes - self.min_corner) / self.bb_size
        # expand each dimension in plus and minus direction to create overlap between sub-regions
        expand_len = 0.5 / splits * overlap_factor

        self.zones = {}
        for z in range(splits):
            z_min = z / splits - expand_len
            z_max = (z + 1) / splits + expand_len
            # index of points in coords_box within the z-range
            overlapping_z_idx = np.logical_and(coords_box[:, 0, 2] <= z_max, coords_box[:, 1, 2] >= z_min).nonzero()[0]
            # the points in the z-range
            overlapping_z_pts = coords_box[overlapping_z_idx,:,:]
            for y in range(splits):
                y_min = y / splits - expand_len
                y_max = (y + 1) / splits + expand_len
                # index of points in overlapping_z_pts within the y-range
                overlapping_y_idx = np.logical_and(overlapping_z_pts[:, 0, 1] <= y_max,
                                                   overlapping_z_pts[:, 1, 1] >= y_min).nonzero()[0]
                # index of points in coords_box
                overlapping_y_box_idx = overlapping_z_idx[overlapping_y_idx]
                # the points in the y-range (and also the z-range)
                overlapping_y_pts = overlapping_z_pts[overlapping_y_idx,:,:]
                for x in range(splits):
                    x_min = x / splits - expand_len
                    x_max = (x + 1) / splits + expand_len
                    overlapping_x_idx = np.logical_and(overlapping_y_pts[:, 0, 0] <= x_max,
                                                       overlapping_y_pts[:, 1, 0] >= x_min).nonzero()[0]
                    overlapping_x_box_idx = overlapping_y_box_idx[overlapping_x_idx]
                    self.zones[(x, y, z)] = overlapping_x_box_idx

    def get_near(self, pt):
        """
        Find triangles near a point. This function is guaranteed to return all triangles whose axis-aligned bou nding
        box (AABB) encloses the point.
        """
        # Get bin numbers
        zone = ((pt - self.min_corner) / self.bb_size) * self.splits
        # Check for spilling over 10.0
        z0 = min(int(zone[0]), 9)
        z1 = min(int(zone[1]), 9)
        z2 = min(int(zone[2]), 9)
        return self.zones[(z0, z1, z2)]
