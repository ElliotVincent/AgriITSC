from src.model import upssits, utae

def get_model(config):
    if config.model == "upssitsv2":
        model = upssits.UPSSITSV2(
            input_dim=10,
            input_size=config.input_size,
            num_classes=config.num_classes,
            app_map=config.app_map,
            temp_int=config.temp_int,
            time_trans=config.time_trans,
            median_filtering=config.median_filtering,
            selected_class=config.selected_class,
            supervised_double=config.supervised_double,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            timestep=config.timestep,
            sigma=config.sigma,
        )
        return model
    elif config.model == "utae":
        model = utae.UTAE(
            input_dim=10,
            encoder_widths=config.encoder_widths,
            decoder_widths=config.decoder_widths,
            out_conv=config.out_conv,
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            agg_mode=config.agg_mode,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            encoder=False,
            return_maps=False,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
        )
        return model
    else:
        raise NotImplementedError


def get_proto_init(num_prototypes, num_steps, input_dim, init_type='constant'):
    """
    init_types: constant, class_means, random, samples
    """
    return None