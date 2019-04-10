Abstract.

Surface inspection systems are an important application domain  for  computer  vision,  as  they  are  used  for  defect  detection  and classification in the manufacturing industry.

Existing systems use hand-crafted  features  which  require  extensive  domain  knowledge  to  create.

Even  though  Convolutional  neural  networks  (CNNs)  have  proven  successful in many large-scale challenges, industrial inspection systems have yet barely realized their potential due to two significant challenges: real-time processing speed requirements and specialized narrow domain-specific datasets which are sometimes limited in size.

In this paper, we propose CNN models that are specifically designed to handle capacity and real-time speed requirements of surface inspection systems.  

To train and evaluate our network models, we created a surface image dataset containing more  than  22000  labeled  images  with  many  types  of  surface  materials and achieved 98.0% accuracy in binary defect classification. 

To solve the class imbalance problem in our datasets, we introduce neural data augmentation  methods  which  are  also  applicable  to  similar  domains  that suffer from the same problem. Our results show that deep learning base methods are feasible to be used in surface inspection systems and outperform traditional methods in accuracy and inference time by considerable margins.

Keywords: convolutional neural networks, image classification, neural data augmentation







表面检测系统是计算机视觉的一个重要应用领域，在制造业中用于缺陷检测和分类。现有的系统使用手工制作的特性，这需要广泛的领域知识来创建。尽管卷积神经网络(CNNs)在许多大规模的挑战中被证明是成功的，但是工业检测系统还没有充分发挥其潜力，这主要是由于两个重大的挑战:实时处理速度要求和特定领域的特殊数据集有时在大小上受到限制。在本文中，我们提出了专门针对地面检测系统的容量和实时速度要求而设计的CNN模型。为了训练和评估我们的网络模型，我们创建了一个表面图像数据集，其中包含超过22000张带有多种表面材料标签的图像，并且在二值缺陷分类中达到了98.0%的准确率。为了解决数据集中的类不平衡问题，我们引入了神经数据增强方法，该方法同样适用于存在相同问题的类似领域。结果表明，深度学习基方法在曲面检测系统中是可行的，在精度和推理时间上均优于传统方法。