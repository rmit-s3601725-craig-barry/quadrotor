import config as cfg
import multiprocessing as mp

"""
    Main experiment script. We pull the experiment config parameters from file and then fire up a bunch of
    instances, with each one running in its own thread (no communication between different instances). This 
    lets us train multiple algorithms on the same environment at the same time. The trainer logs all of the 
    data and saves it to the correct location, after which we can plot it all out using matplotlib or R. 

    -- Sean Morrison, 2018
"""

# TODO:
# Implement a variant of ACER
# Port OffPAC
# Improve FMIS model learning
# Check DDPG noise makes sense after warmup ends


# grab experiment environment and list of algorithms
env_name = cfg.exp["env"]
algs = cfg.exp["algs"]
processes = []

# start training processes
def main(env_name, algs):
    try :
        for alg in algs:
            p = mp.Process(target=make, args=(env_name, alg))
            processes.append(p)
            p.start()

        # Wait for processes to finish
        [p.join() for p in processes];

    #Listens for ctrl+c input
    except KeyboardInterrupt:
        terminate(processes);

# stops the training process
def terminate(processes):
    print('Terminating all training processes');
    #Iterate over each process and terminate them
    for pro in processes:
        pro.terminate();


def make(env_name, alg):
    """
        Builds and returns an instance of a trainer for a given algorithm. All of these algorithms have been
        validated on basic tasks and shown to learn. You can edit the training parameters in the config file
        found in the trainers folder. 
    """

    if alg == "cem":
        params = cfg.cem
        import trainers.cem as cem_trainer
        return cem_trainer.Trainer(env_name, params)
    if alg == "ddpg":
        params = cfg.ddpg
        import trainers.ddpg as ddpg_trainer
        return ddpg_trainer.Trainer(env_name, params)
    if alg == "expected":
        params = cfg.expected
        import trainers.expected as expected_trainer
        return expected_trainer.Trainer(env_name, params)
    if alg == "fmis":
        params = cfg.fmis
        import trainers.fmis as fmis_trainer
        return fmis_trainer.Trainer(env_name, params)
    if alg == "gae":
        params = cfg.gae
        import trainers.gae as gae_trainer
        return gae_trainer.Trainer(env_name, params)
    if alg == "offpac":
        params = cfg.offpac
        import trainers.offpac as offpac_trainer
        return offpac_trainer.Trainer(env_name, params)
    if alg == "ppo":
        params = cfg.ppo
        import trainers.ppo as ppo_trainer
        return ppo_trainer.Trainer(env_name, params)
    if alg == "sw_scv":
        params = cfg.sw
        import trainers.sw_scv as sw_trainer
        return sw_trainer.Trainer(env_name, params)
    if alg == "trpo":
        params = cfg.trpo
        import trainers.trpo as trpo_trainer
        return trpo_trainer.Trainer(env_name, params)
    

if __name__ == "__main__":
    main(env_name, algs)
    