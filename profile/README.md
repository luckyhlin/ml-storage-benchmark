# cifar-10
To process the original data for plotting, we may need the following code

```bash
grep 'cifar-10-batches-py/' profile_cifar-10_dataload.log > profile_cifar-10_read_dataset.log
grep 'read' profile_cifar-10_read_dataset.log > read.log
```

We may also need to remove lines with `.meta` (which was done with the help of GPT) 
