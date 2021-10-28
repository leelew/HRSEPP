from src.data.ops.preprocesser import RawSMAPPreprocesser


class RawDataLoader():
    def __init__(self,
                 raw_data_path,
                 aux,
                 out_path,
                 save_sm_path,
                 save_forcing_path,
                 begin_date,
                 end_date,
                 lat_lower=-90,
                 lat_upper=90,
                 lon_left=-180,
                 lon_right=180,
                 save=True) -> None:
        self.sm_reader = RawSMAPPreprocesser(raw_data_path=raw_data_path,
                                             aux=aux,
                                             out_path=out_path,
                                             save_path=save_sm_path,
                                             begin_date=begin_date,
                                             end_date=end_date,
                                             var_name=['SSM'],
                                             var_list=['sm_surface'],
                                             lat_lower=lat_lower,
                                             lat_upper=lat_upper,
                                             lon_left=lon_left,
                                             lon_right=lon_right,
                                             save=True)
        self.forcing_reader = RawSMAPPreprocesser(
            raw_data_path=raw_data_path,
            aux=aux,
            out_path=out_path,
            begin_date=begin_date,
            save_path=save_forcing_path,
            end_date=end_date,
            var_name=['forcing'],
            var_list=[
                'precipitation_total_surface_flux',
                'radiation_longwave_absorbed_flux',
                'radiation_shortwave_downward_flux',
                'specific_humidity_lowatmmodlay', 'surface_pressure',
                'surface_temp', 'windspeed_lowatmmodlay'
            ],
            lat_lower=lat_lower,
            lat_upper=lat_upper,
            lon_left=lon_left,
            lon_right=lon_right,
            save=True)

    def __call__(self):

        forcing = self.forcing_reader()
        ssm = self.sm_reader()

        return forcing, ssm
