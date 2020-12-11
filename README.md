# CS 591 Assignment 4 
Name: Jinghao Ye

BU ID: U67873405

Email: yjinghao@bu.edu

* Step1: Create a cluster of VMs on MOC. For this step, I follow the OpenStack Tutorial https://docs.massopen.cloud/en/latest/openstack/OpenStack-Tutorial-Index.html
  * Set up a Private Key 
  * Create a Router
  * Create s Security Group
  * Create a key pair 
  * Launch a VM 
  * Assign a Floating IP 
  * SSH to Cloud VM
    To connect the my head vm, I open my terminal and run the command `ssh ubuntu@128.31.27.227`
    and run `ssh ubuntu@192.168.100.49` to connect to my worker node
  
  
* Step2: Deploy Ray on Cluster
  * In my head VM I first run `Ray start --head`; then in the worker VM I run ` ray start --address='192.168.100.161:6379' --redis-password='5241590000000000'
  * In the code change `ray.init()` to `ray.init(address="auto")`
  
* Step3: Set up Jaeger
  * log out head VM and then in the terminal run `ssh -A ubuntu@128.31.27.227 -L 16687:localhost:16686`
  * First I use `scp -r /Users/jinghao/assign4_cluster ubuntu@128.31.27.227:/home/ubuntu` upload my script file to head VM and in head VM run`scp -r /home/ubuntu/assign4_cluster ubuntu@192.168.100.49:/home/ubuntu` upload code to worker node. I also upload a folder name "yml" to the head VM. The folder contain the docker-compose.yml file which is used to configure jaeger's services.
  
  * Next I run docker `pull docker.io/cassandra` and `pull docker.io/cassandra`
  * In both VMs, I run `sudo docker run -d --name jaeger-agent \ -p 5775:5775/udp \ -p 6831:6831/udp \ -p 6832:6832/udp \ -p 5778:5778 \ --restart=always \ jaegertracing/jaeger-agent --reporter.grpc.host-port=192.168.100.161:14250`
 
* Step4: Run and trace the code
  * In the head vm, first change the current directory to the folder containing the code `cd assign4_cluster/` and open ipython by run `ipython`
  * run `%run assignment_4.py`
  * open the website and go to `localhost:16687` to see the trace
