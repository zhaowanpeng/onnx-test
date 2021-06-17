### 1、将pytorch模型转化为.onnx格式

```python

model = Model()#.cuda
#weight=torch.load("weight.pth")	#方法1
#model.load_state_dict(weight)		#方法1
#model = torch.load("model.pth")	#方法2

dummy_input = torch.randn(10, 3, 224, 224, device='cuda')#(b,c,w,h)

# Export the model to an ONNX file

output = torch_onnx.export(model, dummy_input, "hello.onnx", verbose=True)
print("Export of torch_model.onnx complete!")
```

### 2、pytorch输出pth文件

```python
#保存权重
torch.save(model.state_dict(),"weight.pth")
#保存模型加权重
model=Model()
weight=torch.load("weight.pth")
model.load_state_dict(weight)
torch.save(model,"model.pth")

#加载整个模型
model = torch.load("model.pth")
```

### 3、参数

```python
model	# (torch.nn.Module) – 要导出的模型.
args	# (tuple of arguments) – 模型的输入, 任何非Tensor参数都将硬编码到导出的模型中；任何Tensor参数都将成为导出的模型的输入，并按照他们在args中出现的顺序输入。因为export运行模型，所以我们需要提供一个输入张量x。只要是正确的类型和大小，其中的值就可以是随机的。请注意，除非指定为动态轴，否则输入尺寸将在导出的ONNX图形中固定为所有输入尺寸。在此示例中，我们使用输入batch_size 1导出模型，但随后dynamic_axes 在torch.onnx.export()。因此，导出的模型将接受大小为[batch_size，3、100、100]的输入，其中batch_size可以是可变的。
export_params	#(bool, default True) – 如果指定为True或默认, 参数也会被导出. 如果你要导出一个没训练过的就设为 False.
verbose			#(bool, default False) - 如果指定，我们将打印出一个导出轨迹的调试描述。
training		#(bool, default False) - 在训练模式下导出模型。目前，ONNX导出的模型只是为了做推断，所以你通常不需要将其设置为True。
input_names 	#(list of strings, default empty list) – 按顺序分配名称到图中的输入节点
output_names	#list of strings, default empty list) –按顺序分配名称到图中的输出节点
dynamic_axes	# {‘input’ : {0 : ‘batch_size’}, ‘output’ : {0 : ‘batch_size’}}) # variable lenght axes
```

### 

