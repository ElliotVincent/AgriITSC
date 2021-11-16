from src.model import upssits

def get_model(config):
    if config.model == "upssits":
        model = upssits.UPSSITS(
            input_dim=10,
            input_size=(128, 128),
            encoder_widths=config.encoder_widths,
            str_conv_k=config.str_conv_k,
            str_conv_s=config.str_conv_s,
            str_conv_p=config.str_conv_p,
            encoder_norm=config.encoder_norm,
            n_head=config.n_head,
            d_model=config.d_model,
            d_k=config.d_k,
            pad_value=config.pad_value,
            padding_mode=config.padding_mode,
            constant_map=config.constant_intensity_map,
        )
        return model
    else:
        raise NotImplementedError
