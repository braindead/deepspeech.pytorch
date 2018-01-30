import torch
from model import DeepSpeech


if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                        help='path to model file created by training')
    parser.add_argument('--activations', default='hardtanh', help='New activation to use')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    print("Original Activations: ", package.get('activations', 'not_set'))

    package['activations'] = args.activations
    model = DeepSpeech.load_model_package(package)

    if model._activations is not None:
        print("New Activations: ", model._activations)
    else:
        raise RuntimeError("Could not update activations")

    parameters = model.parameters()
    optimizer = torch.optim.Adam(parameters)
    epoch = package.get('epoch', None)
    if epoch:
        epoch -= 1 # indexed from 0
    
    output_name = args.model_path.split(".")[0] + "_" + args.activations
    output_path = output_name + "." + ".".join(args.model_path.split(".")[1:])

    torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, 
        iteration=package.get('iteration', None), loss_results=package.get('loss_results', None), 
        wer_results=package.get('wer_results', None), cer_results=package.get('cer_results', None),
        avg_loss=package.get('avg_loss', None), meta=package.get('meta', None)), output_path)
    print("Saved updated model to {}".format(output_path))
