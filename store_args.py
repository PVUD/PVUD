class Arguments:
    experiment_root = ""
    train_set = ""
    image_root = ""
    query_dataset = ""
    gallery_dataset = ""
    train_dataset = ""
    excluder = ""
    resume = False
    model_name ='resnet_v1_50'
    embedding_dim =128
    initial_checkpoint=None
    batch_p=32
    batch_k=4
    net_input_height=256
    net_input_width=128
    pre_crop_height=288
    pre_crop_width=144
    loading_threads=8
    alpha1=0.05
    alpha2=0.5
    alpha3=0.5
    metric='euclidean'
    learning_rate=3e-4
    train_iterations=25000
    decay_start_iteration=1000
    checkpoint_frequency=5000
    flip_augment=True
    crop_augment=True
    detailed_logs=False
    checkpoint=None
    batch_size=256
    filename=None
    output_name=""
    aggregator='mean'
    selector='mean'
    dataset=''

    def save_args(self, args):
        for key, value in args.__dict__.items():
            self.__dict__[key] = value