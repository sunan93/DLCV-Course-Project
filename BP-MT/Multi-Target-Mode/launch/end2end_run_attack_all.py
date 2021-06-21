import sys
import subprocess

gpu = int(sys.argv[1])
mode = sys.argv[2]
net_repeat = sys.argv[3]
target_num = sys.argv[4]
#eval_poisons_root = sys.argv[5]
assert mode in ['convex', 'mean']
assert gpu >= 0 and gpu <= 15

ids = [1, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19] # we only attack those cars that we already have a good accuracy for the cifar models!
#ids = [15, 16, 17, 19] # we only attack those cars that we already have a good accuracy for the cifar models!
#ids = ids[gpu:gpu+1]
for i in ids:

    
    print("Original Code")
    # original code 
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, False, False, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    
    
    """
    print("Original Code")
    # original code 
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, False, False, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    """

    """ 
    print("Multi layer Code")
    # Multi layer code 
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, False, False, True)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    """
    
    print("Multi layer Code")
    # Multi layer code 
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, False, False, True)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    
    
    
    print("CE loss Code")
    # CE loss code
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, True, False, False, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    
    """
    print("CE loss Code")
    # CE loss code
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, True, False, False, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    """

 
    
    # MAX loss code
    print("Max Loss Code")
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, True, False, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    """
    print("Max Loss Code")
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, True, False, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    """


    
    # Feature reg code 
    print("Feature Reg loss Code")
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, False, True, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    """
    # Feature reg code 
    print("Feature Reg loss Code")
    cmd = 'bash launch/attack-end2end-12.sh {} {} {} {} {} {} {} {} {}'.format(gpu, mode, i, net_repeat, target_num, False, False, True, False)
    print(cmd, cmd.split())
    subprocess.run(cmd.split())
    """


    #cmd = 'bash launch/eval-end2end.sh {} {} {} {}'.format(gpu, i, eval_poisons_root, target_num)
    #print(cmd)
    #subprocess.run(cmd.split())
    i += 1
