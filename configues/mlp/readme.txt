below configue keys are used for cnn: 
    
    
type:'str'    key:"models_dir"                ex: "./models",                   ---> directory for saving the model
type:'str'    key:"logs_dir"                  ex: "./logs",                     ---> log for visualizing with tensorboard
type:'str'    key:"env"                       ex: "CustomHopper-target-v0",     ---> environemnt that we want to use for training
type:'str'    key:"alg"                       ex: "ppo",                        ---> algorithm we use for training ['ppo']
type:'str'    key:"obs"                       ex: "mlp",                        ---> type of feature extraction we want to have for optimization ['cnn', 'mlp']
type:'str'    key:"policy"                    ex: "MlpPolicy",                  ---> policy that we use for training ['MlpPolicy', 'CnnPolicy']
type:'bool'   key:"ignore_warnings"           ex: true,                         ---> ignoring warnings that we dont want to visualize
type:'bool'   key:"print"                     ex: true,                         ---> if the value is true, we print parameters for each model
type:'bool'   key:"custom_arch"               ex: false,                         ---> if true we use custom architectures for feature extraction
type:'int'    key:"time_steps"                ex: 4000,                         ---> number of time steps
    

type:'bool'   key:"domain_randomization"      ex: true,                         ---> if true we randomize the domains
type:'list'    key:"chosen_domain"            ex: "grid",                       ---> it can be a list of domains : [[[0.5, 4]], [[1, 3.5]]] or 'grid' which we use a grid search for finding the optimum distribution
type:'int'    key:"n_distributions"           ex: 1,                            ---> if it's equal to 1, we randomize the all three links with one distributions.
    

type:list['float']    key:"n_steps"                   ex: [2048],                       ---> number of steps before updating the network
type:list['float']    key:"learning_rate"             ex: [0.0004215440315139262],      ---> learning rate for ppo
type:list['float']    key:"gamma"                     ex: [0.9917611787616847],         ---> discount factor for optimizing ppo
type:list['float']    key:"clip_range"                ex: [0.12587511822701497],        ---> clip range during optimization the ppo
type:list['float']    key:"ent_coef"                  ex: [0],                          ---> entropy coefficent for ppo
type:list['float']    key:"gae_lambda"                ex: [0.959132778841134]           ---> Factor for trade-off of bias vs variance for GAE