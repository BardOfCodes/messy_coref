"""
Given a JSON file with run specifications, create job scripts, and launch jobs.
"""
import os
import yaml
from CSG.utils.train_utils import arg_parser
from string import Template
from datetime import datetime

TEMPLATE_FILE = "scripts/job_config/ccv_job.template"
JOB_PATH = "scripts/auto_scripts/"
ALL_JOB_FILE = "run_all.sh"
def main():
    now = datetime.now()
    date_string = now.strftime("%Y_%m_%d_%H")
    
    args = arg_parser.parse_args()
    # load content from a fixed json file.
    content = open(args.job_desc, 'r')
    content = yaml.load(content, Loader=yaml.FullLoader)
    
    template_file = open(TEMPLATE_FILE, 'r')
    src = Template(template_file.read())
    # write content to a bash script
    run_file_list = []
    for job in content['jobs']:
        # For each create a job file and run.
        for key, item in content['defaults'].items():
            if key not in job.keys():
                job[key] = item
        # Create the args
        job['args'] = " ".join(job['pre_args'])
        # Update the output file:
        job['job_file_name'] = "_".join([date_string, job['job_file_name']])
        
        result = src.substitute(job)
        
        job_file = os.path.join(JOB_PATH, job['job_file_name'])
        with open(job_file, 'w') as f:
            f.write(result)
        print("saved file %s" % job_file)
        
        # create the all job script
        run_file_list.append(job_file)
    
    all_job_file = os.path.join(JOB_PATH, ALL_JOB_FILE)
    with open(all_job_file, 'w') as f:
        for filename in run_file_list:
            f.write("sbatch %s \n" % filename)
    print("Saved all file running script at %s" % all_job_file)


if __name__ == "__main__":
    main()