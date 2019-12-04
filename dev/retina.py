import numpy as np
import matplotlib.pyplot as plt

import SLIP
from LogGabor import LogGabor

pe = {'N_image': 100, 'seed': None, 'N_X': 512, 'N_Y': 512, 'noise':
            0.1, 'do_mask': True, 'mask_exponent': 3.0, 'do_whitening': True,
              'white_name_database': 'kodakdb', 'white_n_learning': 0, 'white_N':
                  0.07, 'white_N_0': 0.0, 'white_f_0': 0.4, 'white_alpha': 1.4,
              'white_steepness': 4.0, 'white_recompute': False, 'base_levels':
                  1.618, 'n_theta': 24, 'B_sf': 0.4, 'B_theta': 0.17453277777777776,
              'use_cache': True, 'figpath': 'results', 'edgefigpath':
                  'results/edges', 'matpath': 'cache_dir', 'edgematpath':
                  'cache_dir/edges', 'datapath': 'database/', 'ext': '.pdf', 'figsize':
                  14.0, 'formats': ['pdf', 'png', 'jpg'], 'dpi': 450, 'verbose': 0}

class RetinaWhiten:
    def __init__(self, args, resize=False):
        self.N_X = args.N_X
        self.N_Y = args.N_Y
        self.resize = resize
        self.whit = SLIP.Image(pe=pe)
        self.whit.set_size((self.N_X, self.N_Y))
        # https://github.com/bicv/SLIP/blob/master/SLIP/SLIP.py#L611
        self.K_whitening = self.whit.whitening_filt()
    def __call__(self, pixel_fullfield, zoomW=None):
        if self.resize : # utilise pour WhereResize
            N_X2 = pixel_fullfield[1]
            N_Y2 = pixel_fullfield[2]
            pixel_fullfield = np.array(pixel_fullfield[0])
            self.whit.set_size((N_X2, N_Y2))
            self.K_whitening = self.whit.whitening_filt()
            fullfield = (pixel_fullfield - pixel_fullfield.min()) / (pixel_fullfield.max() - pixel_fullfield.min())
            fullfield = self.whit.FTfilter(fullfield, self.K_whitening)
        else : # utilise pour le reste
            fullfield = (pixel_fullfield - pixel_fullfield.min()) / (pixel_fullfield.max() - pixel_fullfield.min())
            fullfield = self.whit.FTfilter(fullfield, self.K_whitening)
        fullfield = np.array(fullfield)
        #print("RetinaWhiten ok")
        #fullfield = fullfield.flatten() # pour passer en vecteur # a decommenter si RetinaWhiten est la derniere transformation effectuee
        return fullfield

class Retina:
    """ Class implementing the retina transform
    """
    def __init__(self, args):

        self.args = args

        self.N_X = args.N_X
        self.N_Y = args.N_Y

        self.N_theta = args.N_theta
        self.N_azimuth = args.N_azimuth
        self.N_eccentricity = args.N_eccentricity
        self.N_phase = args.N_phase
        self.feature_vector_size = self.N_theta * self.N_azimuth * self.N_eccentricity * self.N_phase

        # !!?? Magic numbers !!??
        self.rho = 1.41
        self.ecc_max = .8  # self.args.ecc_max
        self.sf_0_r = 0.03  # self.args.sf_0_r
        self.sf_0_max = 0.45
        self.B_theta = np.pi / self.N_theta / 2  # self.args.B_theta
        self.B_sf = .4

        self.init_retina_transform()
        self.init_grid()

    def init_grid(self):
        delta = 1. / self.N_azimuth
        self.log_r_grid, self.theta_grid = np.meshgrid(np.linspace(0, 1, self.N_eccentricity + 1), \
                                                       np.linspace(-np.pi * (.5 + delta), np.pi * (1.5 - delta), self.N_azimuth + 1))

    def get_suffix(self):
        suffix = '_{}_{}'.format(self.N_theta, self.N_azimuth)
        suffix += '_{}_{}'.format(self.N_eccentricity, self.N_phase)
        suffix += '_{}_{}'.format(self.args.rho, self.N_pic)
        return suffix

    def init_retina_transform(self):  # ***
        if self.args.verbose: print('Creation du dictionnaire de filtres en cours...')
        self.retina_dico = {}
        pe = {'N_image': 100, 'seed': None, 'N_X': 512, 'N_Y': 512, 'noise':
            0.1, 'do_mask': True, 'mask_exponent': 3.0, 'do_whitening': True,
              'white_name_database': 'kodakdb', 'white_n_learning': 0, 'white_N':
                  0.07, 'white_N_0': 0.0, 'white_f_0': 0.4, 'white_alpha': 1.4,
              'white_steepness': 4.0, 'white_recompute': False, 'base_levels':
                  1.618, 'n_theta': 24, 'B_sf': 0.4, 'B_theta': 0.17453277777777776,
              'use_cache': True, 'figpath': 'results', 'edgefigpath':
                  'results/edges', 'matpath': 'cache_dir', 'edgematpath':
                  'cache_dir/edges', 'datapath': 'database/', 'ext': '.pdf', 'figsize':
                  14.0, 'formats': ['pdf', 'png', 'jpg'], 'dpi': 450, 'verbose': 0}
        lg = LogGabor(pe=pe)
        for i_theta in range(self.N_theta):
            self.retina_dico[i_theta] = {}
            for i_phase in range(self.N_phase):
                self.retina_dico[i_theta][i_phase] = {}
                for i_eccentricity in range(self.N_eccentricity):
                    self.retina_dico[i_theta][i_phase][i_eccentricity] = self.local_filter(i_theta, i_phase, i_eccentricity, lg)
        if self.args.verbose: print("Dico cree")
        if self.args.verbose: print("len finale", len(self.retina_dico), len(self.retina_dico[0]),
                                    len(self.retina_dico[0][0]), len(self.retina_dico[0][0][0]))

    def local_filter(self, i_theta, i_phase, i_eccentricity, lg):

        ecc = self.ecc_max * (1 / self.rho) ** (self.N_eccentricity - i_eccentricity)
        theta_ref = i_theta * np.pi / self.N_theta
        sf_0 = 0.5 * self.sf_0_r / ecc
        sf_0 = np.min((sf_0, self.sf_0_max))

        dimension_filtre = int(1 / sf_0 * 2)
        if dimension_filtre % 2 == 1:
            dimension_filtre += 1
        # print("dimension_filtre", dimension_filtre)
        lg.set_size((dimension_filtre, dimension_filtre))

        params = {'sf_0': sf_0,
                  'B_sf': self.B_sf,
                  'theta': theta_ref,
                  'B_theta': self.B_theta}
        phase = i_phase * np.pi / 2
        return lg.normalize(lg.invert(lg.loggabor(dimension_filtre // 2, dimension_filtre // 2, **params) * np.exp(-1j * phase))).ravel()

    def transform_dico(self, pixel_fullfield):
        fullfield_dot_retina_dico = np.zeros(self.N_theta * self.N_azimuth * self.N_eccentricity * self.N_phase)

        N_X, N_Y = self.N_X, self.N_Y

        indice = 0
        for i_theta in range(self.N_theta):
            for i_phase in range(self.N_phase):
                for i_eccentricity in range(self.N_eccentricity):
                    fenetre_filtre = self.retina_dico[i_theta][i_phase][i_eccentricity]
                    dimension_filtre = int(fenetre_filtre.shape[0] ** (1 / 2))
                    fenetre_filtre = fenetre_filtre.reshape((dimension_filtre, dimension_filtre))
                    for i_azimuth in range(self.N_azimuth):

                        # ecc = ecc_max * (1 / self.args.rho) ** (self.args.N_eccentricity - i_eccentricity)
                        ecc = self.ecc_max * (1 / self.rho) ** ((self.args.N_eccentricity - i_eccentricity) / 3)
                        # /3 ajoute sinon on obtient les memes coordonnees x et y pour environ la moitie des filtres crees 12/07

                        N_min = min(N_X, N_Y)
                        r = np.sqrt(N_min ** 2 + N_min ** 2) / 2 * ecc #- 30  # radius
                        # r = np.sqrt(N_X ** 2 + N_Y ** 2) / 2 * ecc - 30 # radius
                        psi = (i_azimuth + (i_eccentricity % 2) * .5) * np.pi * 2 / self.args.N_azimuth
                        x = int(N_X / 2 + r * np.cos(psi))
                        y = int(N_Y / 2 + r * np.sin(psi))

                        half_width = dimension_filtre // 2
                        x_min = max(int(x - half_width), 0)
                        x_crop_left = min(0, x_min - int(x - half_width))
                        x_max = min(int(x + half_width), N_X)
                        x_crop_right = max(0, int(x + half_width) - x_max)
                        y_min = max(int(y - half_width), 0)
                        y_crop_left = min(0, y_min - int(y - half_width))
                        y_max = min(int(y + half_width), N_Y)
                        y_crop_right = max(0, int(y + half_width) - y_max)


                        fenetre_image = pixel_fullfield[x_min:x_max, y_min:y_max]
                        fenetre_filtre_crop = fenetre_filtre[x_crop_left:-x_crop_right, y_crop_left, -y_crop_right]
                        print(fenetre_image.shape, fenetre_filtre_crop.shape)

                        a = np.dot(np.ravel(fenetre_filtre_crop), np.ravel(fenetre_image))
                        fullfield_dot_retina_dico[indice] = a

                        indice += 1

        return pixel_fullfield, fullfield_dot_retina_dico