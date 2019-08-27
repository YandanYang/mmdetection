#load model

##tools/train.py
```
model = build_detector(
     -->cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
##mmdet/model/builder.py
def build_detector(cfg, train_cfg=None, test_cfg=None):
 -->return build(cfg, DETECTORS, dict(train_cfg=train_cfg, test_cfg=test_cfg))
def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
         -->build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)
```   
##mmdet/utils/registry.py
```
def build_from_cfg(cfg, registry, default_args=None):
    ...
    obj_type = args.pop('type') #RPN
    if mmcv.is_str(obj_type):
        obj_cls = registry.get(obj_type) #mmdet/models/detector/rpn.py -->RPN()
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif inspect.isclass(obj_type):
        obj_cls = obj_type
    else:
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)
    return obj_cls(**args)　#RPN(**args)
```
#train model
##tools/train.py
```
train_detector(model,
              datasets,
              cfg,
              distributed=distributed,
              validate=args.validate,
              logger=logger)
              
```
##mmdet/apis/train.py              
```
def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   logger=None):
    if logger is None:
        logger = get_root_logger(cfg.log_level)

    # start training
    if distributed:
        _dist_train(model, dataset, cfg, validate=validate)
    else:
     -->_non_dist_train(model, dataset, cfg, validate=validate)
     
def _non_dist_train(model, dataset, cfg, validate=False):
    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False) for ds in dataset
    ]
    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)
    -->runner = Runner(model, batch_processor, optimizer, cfg.work_dir,
                    cfg.log_level)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=False)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
 -->runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
```

 ##mmcv/runner/runner.py
```
-->run()
 -->def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(data_loader)
        self.call_hook('before_train_epoch')
        for i, data_batch in enumerate(data_loader):
            self._inner_iter = i
            self.call_hook('before_train_iter')
         -->outputs = self.batch_processor(
                self.model, data_batch, train_mode=True, **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('batch_processor() must return a dict')
            if 'log_vars' in outputs:
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
            self.outputs = outputs
         -->self.call_hook('after_train_iter')
            self._iter += 1

        self.call_hook('after_train_epoch')
        self._epoch += 1
```
##mmdet/apis/train.py 
```
def batch_processor(model, data, train_mode):
 -->losses = model(**data)
    loss, log_vars = parse_losses(losses)

    outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
    return outputs
```
##mmdet/models/anchor_heads/rpn_head.py
##mmdet/models/anchor_heads/anchor_head.py
##mmcv/runner/runner.py
```
def call_hook(self, fn_name):
        for hook in self._hooks:
            getattr(hook, fn_name)(self)
```         
##mmcv/runner/runner.py
```
def register_hook(self, hook, priority='NORMAL'):
def build_hook()
```
##mmcv/runner/hooks/optimizer.py
```
class OptimizerHook(Hook):

    def __init__(self, grad_clip=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        clip_grad.clip_grad_norm_(
            filter(lambda p: p.requires_grad, params), **self.grad_clip)

 -->def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            self.clip_grads(runner.model.parameters())
        runner.optimizer.step()
```
