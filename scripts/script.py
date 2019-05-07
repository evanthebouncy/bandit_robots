import os

def make_instance(snapshot_name, instance_name):
    ret = \
    f"""
gcloud compute --project "capgroup" disks create "{instance_name}" --size "100" --zone "us-east1-b" --source-snapshot "{snapshot_name}" --type "pd-standard"

gcloud compute --project=capgroup instances create {instance_name} --zone=us-east1-b --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --maintenance-policy=TERMINATE --service-account=881379218043-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=type=nvidia-tesla-p100,count=1 --tags=http-server,https-server --disk=name=instance-1,device-name=instance-1,mode=rw,boot=yes,auto-delete=yes

    """
    return ret

def run_remote_cmd(folder_path, script_path):
    ret = \
    f"""
gcloud compute ssh --zone=us-east1-b yewenpu@instance-1 --command='cd {folder_path}; git pull; chmod 777 {script_path}; {script_path} '

    """

if __name__ == "__main__":
    snapshot_name = "snapshot-test-pytorch1"
    instance_name = "instance-1"
    print (make_instance(snapshot_name, instance_name))

    folder_path = "/home/yewenpu/bandit_robots"
    script_path = "./scripts/run.sh" 