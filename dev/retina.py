import numpy as np
import matplotlib.pyplot as plt

import SLIP
from LogGabor import LogGabor

pe = {
    # Log-Gabor
    #'base_levels' : 2.,
    'base_levels' : 1.618,
    'n_theta' : 24, # number of (unoriented) angles between 0. radians (included) and np.pi radians (excluded)
    'B_sf' : .4, # 1.5 in Geisler
    'B_theta' : 3.14159/18.,
    }


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
        filename = '../tmp/retina' + self.get_suffix() + '_dico.npy'
        if self.args.verbose: print(filename)
        try:
            self.retina_dico = np.load(filename, allow_pickle=True).item()
            if self.args.verbose: print("Fichier retina_dico charge avec succes")
        except:
            if self.args.verbose: print('Creation du dictionnaire de filtres en cours...')
            self.retina_dico = {}
            lg = LogGabor(pe=pe)
            for i_theta in range(self.N_theta):
                self.retina_dico[i_theta] = {}
                for i_phase in range(self.N_phase):
                    self.retina_dico[i_theta][i_phase] = {}
                    for i_eccentricity in range(self.N_eccentricity):
                        self.retina_dico[i_theta][i_phase][i_eccentricity] = self.local_filter(i_theta, i_phase, i_eccentricity, lg)
            if self.args.verbose: print("Dico cree")
            np.save(filename, self.retina_dico)
            if self.args.verbose: print("len finale", len(self.retina_dico), len(self.retina_dico[0]),
                                        len(self.retina_dico[0][0]), len(self.retina_dico[0][0][0]))
            if self.args.verbose: print("Fichier retina_dico ecrit et sauvegarde avec succes")

    def local_filter(self, i_theta, i_phase, i_eccentricity, lg):
        # rho=1.41, ecc_max=.8,
        # sf_0_max=0.45, sf_0_r=0.03,
        # B_sf=.4, B_theta=np.pi / 12):

        # !!?? Magic numbers !!??
        rho = 1.41
        ecc_max = .8  # self.args.ecc_max
        sf_0_r = 0.03  # self.args.sf_0_r
        sf_0_max = 0.45
        B_theta = np.pi / self.N_theta / 2  # self.args.B_theta
        B_sf = .4

        ecc = ecc_max * (1 / rho) ** (self.N_eccentricity - i_eccentricity)
        theta_ref = i_theta * np.pi / self.N_theta
        sf_0 = 0.5 * sf_0_r / ecc
        sf_0 = np.min((sf_0, sf_0_max))

        dimension_filtre = int(1 / sf_0 * 2)
        if dimension_filtre % 2 == 1:
            dimension_filtre += 1
        # print("dimension_filtre", dimension_filtre)
        lg.set_size((dimension_filtre, dimension_filtre))

        params = {'sf_0': sf_0,
                  'B_sf': B_sf,
                  'theta': theta_ref,
                  'B_theta': B_theta}
        phase = i_phase * np.pi / 2
        return lg.normalize(lg.invert(lg.loggabor(dimension_filtre // 2, dimension_filtre // 2, **params) * np.exp(-1j * phase))).ravel()


