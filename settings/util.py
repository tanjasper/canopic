import json

def load_encoder_settings(json_path):
    with open(json_path, 'r') as fp:
        encoder_json = json.load(fp)
    return encoder_json['encoder_archs'], encoder_json['encoder_settings']

def load_discriminator_settings(json_path):
    with open(json_path, 'r') as fp:
        discriminator_json = json.load(fp)
    return discriminator_json['discriminator_arch'], discriminator_json['discriminator_settings']
