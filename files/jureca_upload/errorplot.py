import pandas as pd
import AnalFunctions as af
path, folder_list, N_runs, b, cross_var, folder_frame, test_str, test_var, test_var2, test_str2, lin_var, T_test_list, sec_test_var, N_ped, fps, mot_frac = af.var_ini()

fps = 16
fps_step = 8
dens_files = pd.read_csv(path + "/density/file_list.csv")
dens_files_ini = pd.read_csv(path + "/density/file_list_ini.csv")
af.error_plot_writer(dens_files,"error_plot.csv",10 * fps,5 * fps,fps_step)
af.error_plot_writer(dens_files,"error_plot5s.csv",5 * fps,5 * fps,fps_step)

af.error_plot_writer(dens_files_ini,"error_plot_ini.csv",0,fps_step,fps_step)
