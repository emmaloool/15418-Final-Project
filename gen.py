#!/usr/bin/env python3

####################################################################
#  A script to test our implementation of cwebp with CUDA support
#  ______________________________________________________________
# 
#  Prior to running this script, users should run:
#
#       cmake -DWEBP_BUILD_CWEBP=ON -DWEBP_BUILD_DWEBP=ON -DWEBP_BUILD_WEBPINFO=ON -
#       DCMAKE_CUDA_FLAGS="-gencode=arch=compute_30,code=sm_30 
#       -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52 
#       -gencode=arch=compute_60,code=sm_60 -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 
#       -Xptxas=-v --ptxas-options=-v" ..
#
#  followed by a standard make command.
####################################################################

import subprocess
import os
import sys
import argparse
import re

project_path = os.path.abspath(".")
build_path = os.path.join(project_path, "build") 

INPUT = os.path.join(project_path, "IMAGES/INPUT") 
OUTPUT = os.path.join(project_path, "IMAGES/OUTPUT") 

# Collects some statistics about trace timings... by just dumping the output into a single file.
runtime_file = "runtime_information.txt"


def get_args():
    # Set up CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')

    # Optional flags
    parser.add_argument('-i', '--iterations', action='store', type=int, help='Number of iterations for testing')
    parser.add_argument('-f', '--file', action='store', help="Path to test a single image")
    parser.add_argument('-o', '--output', action='store', help="Path of alternate file to pipe timings. Default file used is './runtime_information.txt'")
    args = parser.parse_args()

    # Parse and verify arguments

    # Optional argument: alternate file path for dumping timing information. 
    # Default path is set to main directory folder to a file "runtime_information.txt"
    if (not args.output) or (not os.path.isfile(args.output)): 
        args.output = os.path.join(project_path, runtime_file)

    # Optional argument: number of iterations for testing. 
    # Default behavior is to run cwebp once on each image in ./IMAGES/INPUT folder
    if (not args.iterations):
        args.iterations = 1

    # Optional argument: number of iterations for testing. 
    # Default behavior is to run cwebp once on each image in ./IMAGES/INPUT folder
    if (args.file) and (not os.path.isfile(args.file)):
        args.file = None

    # ! TODO: add "lossless" cwebp commandline parameter here

    return args


def run_cwebp(args):
    print(INPUT)
    if not os.path.isdir(INPUT):
        print("Cannot access ../IMAGES/INPUT path.")
        sys.exit(0)

    # Prepare directory to contain image output files
    runtime_files_path = os.path.join(OUTPUT, "runtimes")
    # if not os.path.isdir(runtime_files_path):
    #     os.mkdir(runtime_files_path)

    big_file = os.path.join(project_path, runtime_file)
    bg = open(big_file, 'w+')

    if args.file:
        # Extrace basename from original image file
        file_base = os.path.splitext(args.file)[0]
        file_webp = file_base + '.webp'

        print("#################### Converting {} to {}... ####################\n".format(args.file, file_webp), file=f)

        img_output_file = os.path.join(runtime_files_path, "{}.txt".format(file_base))
        cmd = "{} -lossless {} -o {}".format(os.path.join(build_path, "cwebp"), 
                                            os.path.join(INPUT, args.file), 
                                            os.path.join(OUTPUT, file_webp))


        print("Running iteration 1/1... ")
        print("----------------------------------------------------------------")
        subprocess.run(cmd, stdout=open(img_output_file, 'w+'), stderr=subprocess.STDOUT, shell=True)
        print("----------------------------------------------------------------\n")

    else: 
        # Iterations will be produced i times consecutively for each photo
        for img in os.listdir(INPUT):
            file_base = os.path.splitext(img)[0]
            file_webp = file_base + '.webp'

            img_output_file = os.path.join(runtime_files_path, "{}.txt".format(file_base))
            f = open(img_output_file, 'w+')

            print("#################### Converting {} to {}... ####################\n".format(img, file_webp), file=f)

            cmd = "{} -lossless {} -o {}".format(os.path.join(build_path, "cwebp"), 
                                                os.path.join(INPUT, img), 
                                                os.path.join(OUTPUT, file_webp))
            
            for i in range(0, args.iterations):
                print("Running iteration {}/{}...".format(i, args.iterations), file=f)
                print("----------------------------------------------------------------", file=f)
                f.flush()
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
                print("----------------------------------------------------------------\n", file=f)
                f.flush()
            f.close()

            f = open(img_output_file, 'r')
            bg.write(f.read())
            f.close()


def main():
    args = get_args()
    run_cwebp(args)
    

if __name__ == '__main__':
    main()
