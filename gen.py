#!/usr/bin/env python3

######################################################################################
#  A script to test our implementation of cwebp with CUDA support
#  __________________________________________________________________________________
# 
#  Prior to running this script, users should run:
#
#       cmake -DWEBP_BUILD_CWEBP=ON -DWEBP_BUILD_DWEBP=ON -DWEBP_BUILD_WEBPINFO=ON -
#       DCMAKE_CUDA_FLAGS="-gencode=arch=compute_30,code=sm_30 
#       -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_50,code=sm_50 
#       -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_60,code=sm_60 
#       -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_61,code=compute_61 
#       -Xptxas=-v --ptxas-options=-v" ..
#
#  followed by a standard make command to create the cwebp binaries. 
#  
#  This script relies on the use of 2 different binaries: 
#  one for CUDA's version [cwebp-cuda], one for the standard C's version [cwebp-c]
#  Separate binaries need to be created prior to running this script.
#
######################################################################################

import argparse
import subprocess
import os
import shutil
import csv
import re

project_path = os.path.abspath(".")
build_path = os.path.join(project_path, "build") 
output_path = os.path.join(project_path, "output") 

INPUT = os.path.join(project_path, "INPUT") 
OUTPUT = os.path.join(output_path, "img") 

# Collects some statistics about trace timings... by just dumping the output into a single file.
runtime_path = os.path.join(output_path, "runtime_information.txt")
stats_path = os.path.join(output_path, "stats")
csv_file = os.path.join(output_path, "data.csv")

def get_args():
    # Set up CMD arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')

    # Optional flags
    parser.add_argument('-n', '--iterations', action='store', type=int, help='Number of iterations for testing')
    parser.add_argument('-f', '--file', action='store', help="Path to test a single image")
    parser.add_argument('-t', '--timing_file', action='store', help="Path of alternate file to pipe timings. Default file used is './runtime_information.txt'")
    parser.add_argument('-i', '--image_inputs', action='store', help="Path of directory of image inputs")
    parser.add_argument('-o', '--image_outputs', action='store', help="Path of directory of image inputs")
    args = parser.parse_args()

    # Parse and verify arguments

    # Optional argument: alternate file path for dumping timing information. 
    # Default path is set to main directory folder to a file "runtime_information.txt"
    if (not args.timing_file) or (not os.path.isfile(args.timing_file)): 
        args.timing_file = runtime_path

    # Optional argument: number of iterations for testing. 
    # Default behavior is to run cwebp once on each image in ./IMAGES/INPUT folder
    if not args.iterations:
        args.iterations = 1

    # Optional argument: alternate file path for image input directory
    if not args.image_inputs:
        args.image_inputs = INPUT 

    # Optional argument: alternate file path for image output directory
    if not args.image_outputs:
        args.image_outputs = OUTPUT

    # ! TODO: add "lossless" cwebp commandline parameter here

    return args


def run_image(img_name, big_file, data_writer, args):
    file_base = os.path.splitext(img_name)[0]
    file_webp = file_base + ".webp"

    img_output_file = os.path.join(stats_path, "{}.txt".format(file_base))
    f = open(img_output_file, 'w+')

    # ---------------------------------------------------------------------------------------------
    
    print("#################### cwebp-C: Converting {} to {}... ####################\n".format(img_name, file_webp), file=f)
    f.flush()

    # Run cwebp-C to find image compression efficiency
    # Note that this only needs to be run once, as the value is deterministic 

    c_cmd = "{} -lossless {} -o {}".format(os.path.join(build_path, "cwebp-c"), 
                                        os.path.join(args.image_inputs, img_name), 
                                        os.path.join(args.image_outputs, file_webp))

    subprocess.run(c_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
    f.flush()

    print("\n#################### cwebp-CUDA: Converting {} to {}... ####################\n".format(img_name, file_webp), file=f)
    
    # Run cwebp-cuda for i iterations + write out data to image-specific file

    cuda_cmd = "{} -lossless {} -o {}".format(os.path.join(build_path, "cwebp"), 
                                        os.path.join(args.image_inputs, img_name), 
                                        os.path.join(args.image_outputs, file_webp))

    for i in range(0, args.iterations):
        print("Running cwebp-CUDA, iteration {}/{}...".format(i+1, args.iterations), file=f)
        print("----------------------------------------------------------------", file=f)
        f.flush()
        subprocess.run(cuda_cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
        print("----------------------------------------------------------------\n", file=f)
        f.flush()

    f.close()

    # ---------------------------------------------------------------------------------------------

    f = open(img_output_file, 'r')

    # Copy per-file output to overall file
    file_text = f.read()
    big_file.write(file_text)

    # Copy per-file output data to overall CSV
    cuda_times = re.findall(r'duration_cuda = ([0-9\.]+)', file_text)
    c_times = re.findall(r'duration_C = ([0-9\.]+)', file_text)
    bpp_list = re.findall(r'Lossless-ARGB compressed size: ([0-9\.]+)', file_text)
    cuda_size = bpp_list[1] # remaining instances are from CUDA
    c_size = bpp_list[0] # first bpp in file is C version's

    for j in range(0, args.iterations):
        data_writer.writerow([file_base, cuda_times[j], c_times[j], cuda_size, c_size])

    f.close()
        

def run_CWEBP(args):
    if not os.path.isdir(args.image_inputs):
        print(args.image_inputs)
        print("Cannot access image input path.")
        return

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)

    # Set up directory to contain WebP images
    os.mkdir(args.image_outputs)

    # Set up directory to contain individual image output files
    os.mkdir(stats_path)

    # Set up writing individual image data files to larger file in overall project file directory
    bg = open(runtime_path, 'w+')

    # Set up writing larger CSV file containing data scrapped from individual image data files
    # CSV format:  IMG NAME | CUDA time | C time | CUDA compression ratio | C compression ratio
    csv_path = os.path.join(project_path, csv_file)
    data_file = open(csv_path, "w+")
    data_writer = csv.writer(data_file, delimiter=',')

    if args.file:
        run_image(args.file, bg, data_writer, args)
    else: 
        # Iterations will be produced i times consecutively for each photo
        for img in os.listdir(args.image_inputs):
            run_image(img, bg, data_writer, args)

    # Clean up resources
    bg.close()
    data_file.close()


def main():
    args = get_args()
    run_CWEBP(args)
    

if __name__ == '__main__':
    main()
