#coding=utf-8
import numpy as np
import tensorflow as tf
import os

from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


# 一 参数入口
tf.app.flags.DEFINE_string("ps_hosts", "localhost:2222", "ps hosts") #指定集群中的参数服务器地址
tf.app.flags.DEFINE_string("worker_hosts", "localhost:2223", "worker hosts") # 指定集群中计算服务器地址

tf.app.flags.DEFINE_string("job_name", "worker", "'ps' or'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job") # 指定当前程序的任务
tf.app.flags.DEFINE_integer("num_workers", 1, "Number of workers")

tf.app.flags.DEFINE_string("model_path", "./dis_model", "the path of the model")
tf.app.flags.DEFINE_integer("v",1,"model version")

FLAGS = tf.app.flags.FLAGS


# 生成数据
def load_data(num_samples=100):
    x_data = np.arange(num_samples,step=.1)
    y_data = x_data + 20 * np.sin(x_data/10)
    
    x_data = np.reshape(x_data,(num_samples*10,1))
    y_data = np.reshape(y_data,(num_samples*10,1))
    
    return x_data,y_data


def fit_1():
    
    # server解析
    ps_hosts = FLAGS.ps_hosts.split(",") #ps
    worker_hosts = FLAGS.worker_hosts.split(",") #worker
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts}) #集群配置文件

    server = tf.train.Server(cluster, job_name=FLAGS.job_name,task_index=FLAGS.task_index) # 创建本地的server

    if FLAGS.job_name == "ps": #参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程
        with tf.device("/cpu:0"):    
            server.join()  
    
    elif FLAGS.job_name == "worker": #计算服务器需要执行计算和训练过程
        
        # 自动将所有参数分配到参数服务器上，将计算分配到当前的计算服务器上
        device_setter = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % (FLAGS.task_index),cluster=cluster)
        with tf.device(device_setter):
            
            
            # 构建图
            input_x = tf.placeholder(tf.float32, [None, 1],name="x_real")
            input_y = tf.placeholder(tf.float32,[None,1],name="y_real")
            
            b = tf.Variable(tf.zeros([1]))
            W = tf.Variable(tf.random_uniform([1,1], -0.1, 0.1))
            y = tf.matmul(input_x,W) + b
            
            loss = tf.reduce_mean(tf.square(y - input_y)) # 最小化方差
            optimizer = tf.train.GradientDescentOptimizer(0.0001) # 优化器
            
            global_step = tf.train.get_or_create_global_step()
            train_op = optimizer.minimize(loss, global_step=global_step)
            
            #============图构建完毕====================
            
            init_op = tf.initialize_all_variables()
            saver = tf.train.Saver()
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            logdir="./log/",
                            init_op=init_op,
                            summary_op=None,
                            saver=saver)
    
            #管理深度学习模型的通用功能。
            with sv.prepare_or_wait_for_session(server.target) as sess:
    
                #1 执行迭代过程。在迭代过程中帮助完成初始化，从checkpoint中加载训练过的模型、输出日志并保存模型
                for ecpoch in range(10):
                    data_x,data_y = load_data() # 数据加载
                    result_train_op,ls, step = sess.run([train_op, loss, global_step],feed_dict={input_x: data_x, input_y: data_y}) # 执行运行
                    print("Train ecpoch %d, loss: %f" % (step, ls))
                    
                
                # 保存模型
                sess.graph._unsafe_unfinalize()
                export_path = os.path.join(FLAGS.model_path,str(FLAGS.v)) # 模型保存路径
                builder = tf.saved_model.builder.SavedModelBuilder(export_path) # 构建保存器
                
                #签名信息
                tensor_info_x = tf.saved_model.utils.build_tensor_info(input_x)
                tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
                prediction_signature = (
                        tf.saved_model.signature_def_utils.build_signature_def(
                                inputs={'x': tensor_info_x},
                                outputs={'y': tensor_info_y},
                                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                        )
                )
                
                #添加构建图和变量至构建器中
                builder.add_meta_graph_and_variables(
                        sess, [tf.saved_model.tag_constants.SERVING],
                        signature_def_map={'model_predict':prediction_signature},
                        main_op=tf.tables_initializer(),strip_default_attrs=True
                )
                
                builder.save() #保存模型
                sess.graph.finalize()
                    
            sv.stop()
            
def fit_2():
    
    # server解析
    ps_hosts = FLAGS.ps_hosts.split(",") #ps
    worker_hosts = FLAGS.worker_hosts.split(",") #worker
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts}) #集群配置文件

    server = tf.train.Server(cluster, job_name=FLAGS.job_name,task_index=FLAGS.task_index) # 创建本地的server

    if FLAGS.job_name == "ps": #参数服务器只需要管理TensorFlow中的变量，不需要执行训练的过程
        with tf.device("/cpu:0"):    
            server.join()  
    
    elif FLAGS.job_name == "worker": #计算服务器需要执行计算和训练过程
        
        # 自动将所有参数分配到参数服务器上，将计算分配到当前的计算服务器上
        device_setter = tf.train.replica_device_setter(worker_device="/job:worker/task:%d" % (FLAGS.task_index),cluster=cluster)
        with tf.device(device_setter):
            
            
            # 构建图
            input_x = tf.placeholder(tf.float32, [None, 1],name="x_real")
            input_y = tf.placeholder(tf.float32,[None,1],name="y_real")
            
            b = tf.Variable(tf.zeros([1]))
            W = tf.Variable(tf.random_uniform([1,1], -0.1, 0.1))
            y = tf.matmul(input_x,W) + b
            
            loss = tf.reduce_mean(tf.square(y - input_y)) # 最小化方差
            optimizer = tf.train.GradientDescentOptimizer(0.0001) # 优化器
            
            #The StopAtStepHook handles stopping after running given steps.
            hooks = [tf.train.StopAtStepHook(last_step=200)]  #迭代次数
            global_step = tf.train.get_or_create_global_step()
    
#            train_op = optimizer.minimize(loss, global_step=global_step,aggregation_method=tf.AggregationMethod.ADD_N)
            train_op = optimizer.minimize(loss, global_step=global_step)
            
            #============图构建完毕====================
    
            #管理深度学习模型的通用功能。
            sess_config = tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
            with tf.train.MonitoredTrainingSession(master=server.target,is_chief=(FLAGS.task_index == 0),checkpoint_dir="./checkpoint_dir",hooks=hooks,config=sess_config) as mon_sess:
    
                #1 执行迭代过程。在迭代过程中帮助完成初始化，从checkpoint中加载训练过的模型、输出日志并保存模型
                while not mon_sess.should_stop():
                    
                    data_x,data_y = load_data() # 数据加载
                    
                    result_train_op,ls, step = mon_sess.run([train_op, loss, global_step],feed_dict={input_x: data_x, input_y: data_y}) # 执行运行
                    
                    # 每段时间输出训练信息。不同的计算服务器都会更新全局的训练轮数，所以这里使用
                    #global_step得到在训练中使用过的batch的总数
                    if step % 10 == 0:
                        print("Train step %d, loss: %f" % (step, ls))
                    
                
                # 保存模型
                export_path = os.path.join(FLAGS.model_path,str(FLAGS.v)) # 模型保存路径
                builder = tf.saved_model.builder.SavedModelBuilder(export_path) # 构建保存器
                
                #签名信息
                tensor_info_x = tf.saved_model.utils.build_tensor_info(input_x)
                tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
                prediction_signature = (
                        tf.saved_model.signature_def_utils.build_signature_def(
                                inputs={'x': tensor_info_x},
                                outputs={'y': tensor_info_y},
                                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                        )
                )
                
                #添加构建图和变量至构建器中
                builder.add_meta_graph_and_variables(
                        mon_sess, [tf.saved_model.tag_constants.SERVING],
                        signature_def_map={'model_predict':prediction_signature},
                        main_op=tf.tables_initializer(),strip_default_attrs=True
                )
                
                builder.save() #保存模型
                
            
def client_1():
    """
    测试 API
    :return: 
    """
    # 随机产生 10 条测试数据
    data_size = 10
    data_x,data_y = load_data()
    data_x = data_x[:data_size]

    channel = implementations.insecure_channel('localhost', int(5000))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    # 发送请求
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'inception'
    request.model_spec.signature_name = "model_predict"
    request.inputs["x"].CopyFrom(tf.contrib.util.make_tensor_proto(data_x, shape=[data_x.shape[0], data_x.shape[1]], dtype=tf.float32))
    # 10 秒超时
    res = stub.Predict(request, 10.0)

    print(res.outputs["y"])
            

def main(_):
    
    fit_1()
    
    #fit_2()
    
#    client_1()
            

if __name__ == "__main__":
    
    tf.app.run() #运行TensorFlow程序


















