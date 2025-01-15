## **技能**

### **1. NLP 和 LLM 相关建模及应用**

- 熟练掌握现代 NLP 算法和神经网络架构，具有从研究论文中复现和应用前沿算法的经验和能力。
- 擅长使用 LLM、Mask-LM 及经典模型 NLP 算法，涵盖分类、序列标注、文本生成等多种任务。
- 熟悉通过 vLLM、ONNX、gRPC 等技术构建 NLP 算法的应用。
- 善于根据不同业务需求定制算法，将 NLP 专业能力迁移到新领域或新应用中。

### **2. Java Web 开发（Spring Framework）**

- 深入理解 Spring Framework，包括 Spring Boot、Spring Data、IoC、Bean、Configuration 及 Annotation 等。
- 能够从零开始构建基于 Spring Boot 的 Web 应用，并具有根据源码分析和排查问题能力。

### **3. Python 数据处理与算法调用服务构建**

- 精通 Python，曾开发和参与多个开源项目。
- 熟悉使用 FastAPI、Flask、Peewee、Celery、Pydantic 和 gRPC 。
- 掌握 PyTorch、HuggingFace Transformers 和 ONNX Runtime 等多种机器学习库，并具有模型训练与部署经验。
- 具备且熟悉通过 gRPC、FastAPI 和 Celery 将 NLP 模型部署为服务的能力。

### **4. Devops 自动化构建**

- 具有建立和维护代码库并在此之上进行自动化CI/CD构建系统的部署经验。
- 具有搭建 Maven、NPM、Docker 和 Python 包的私有库，具备实现全自动 CI/CD 自动化构建部署 pipeline 的经验。
- 制定编码规范及构建可复用模块，覆盖 Java、JavaScript、Python 和 Docker 等方面。

### **5. GJB-9001C 标准实施**

- 作为实施及首次标准审核受审人员通过 GJB-9001C 并获得资质证书。
- 在项目中执行 GJB-9001C 标准并主要编写了相应项目文档, 具有实施相关标准的经验和相应的文档编写能力。

---

## **重点项目**

### **1. 文档管理与分析系统**

1. **描述**：基于 OCR 和 NLP 技术的扫描文档管理和分析系统。
3. **角色**: 项目负责人, 构架、后端、算法的主要实施者。
2. **主要贡献**：  
    - 使用 PaddleOCR 并通过微调优化实现并达到 OCR 预期识别精度。
    - 基于 BERT（P-tuning）开发关键信息抽取算法，并后续迁移到 Bloomz-7B 模型并使用 llama.cpp 进行推理，二期项目中使用 LLM 并利用 vLLM 以及 CFG 实现思维链(CoT)提升算法性能。
    - 构建 Elasticsearch 分词插件，使用 NLPIR-ICTCLAS 分词器实现自定义和高效分词索引。
    - 集成 SANE 实现扫描仪控制，开发基于自定义 GRUB2 镜像和 PXE 启动的无盘工作站解决方案。

### **2. 知识图谱构建系统**

1. **描述**: 一个根据给定数据进行半自动化构建知识图的平台, 并支持根据系统内数据进行智能问答、可视化等功能。
2. **角色**: 项目负责人, 构架、后端、算法的主要实施者。
3. **主要贡献**：
    1. **流程设计**：设计完整的知识图谱构建流程，平衡使用需求和当下 LLM 进行知识图谱构建的准确率。
    2. **关键算法**：
        - 利用 LLM-guided workflow 实现思维链(CoT)提升算法性能 并利用基于 CFG 的 JSON 输出获取格式化生成内容。
        - 利用上述 workflow 实现命名实体识别、属性提取、实体消歧和关系抽取算法。
        - 构建混合检索（KNN + BM25）及递归式 RAG 工作流实现基于知识图谱数据检索的智能对话。
4. **技术栈**：  
    - 使用 vLLM 动态加载 LoRA 模型, 实现多个模型共享单一大模型推理。
    - 基于 Spring Boot、Spring Data（JPA、Elasticsearch、Neo4j）、Jackson 和 Swagger 构建 API。

### **3. 情报信息分析与报告辅助生成系统**
1. **角色**: 项目负责人, 构架、后端、算法的主要实施者。
2. **主要贡献**：  
    - 使用 LLM-guided workflow 和模型微调实现多级分类与多标签标注算法。
    - 对 LLM 进行 token 词典扩充和二次训练微调实现小语种机器翻译。
    - 通过 RAG + LLM-guided 思维链（CoT）实现情报分析和报告生成。
    - 利用微调后的 LLM 和 基于 CFG 的结构化输出生成实现文本校对功能。

### **4. NLPIR 工具库（统计方法的 NLP 工具包）**

1. NLPIR 开发库, 包括 Python, Java 和 gRPC,HTTP 调用的分布式 NLPIR 算法调用服务
    - [NLPIR-python](https://github.com/NLPIR-team/nlpir-python) Python package
    - [NLPIR-cloud](https://github.com/NLPIR-team/nlpir-cloud) gRPC,HTTP 调用的分布式 NLPIR 算法调用服务, JWT 鉴权
    - [nlpir-java-client](https://github.com/NLPIR-team/nlpir-java-client) nlpir-cloud 的 Java 封装
2. [elasticsearch-analysis-ictclas](https://github.com/NLPIR-team/elasticsearch-analysis-ictclas) Elasticsearch 分词插件
---

## **开源贡献**

1. **[Llama.cpp](https://github.com/ggerganov/llama.cpp)**: Fixed bug in ALiBi positional encoding ([PR #143](https://github.com/ggerganov/ggml/pull/143)).  
2. **[Pgsync](https://github.com/toluaina/pgsync)**: Add feature to make current path can use and execute config in environment. ([PR #250](https://github.com/toluaina/pgsync/pull/250)).  
3. **[Elasticsearch](https://github.com/elastic/elasticsearch)**: AAdd feature to recognize file’s type with filename based on tika support. ([PR #64389](https://github.com/elastic/elasticsearch/pull/64389)).  
4. **[Jina AI](https://github.com/jina-ai/clip-as-service)**: Add feature to embedding text larger than max length. ([PR #447](https://github.com/jina-ai/clip-as-service/pull/447)).  


---

## **经历**

#### 1. **2010–2014**：北京印刷学院 **计算机科学学士**  
- 毕业论文：《基于贝叶斯理论的万智牌卡牌推荐算法》[点击查看](https://www.jsjkx.com/CN/Y2014/V41/IZ11/72)  

#### 2. **2016~2017** 兼职
- 专利检索和分析系统
    - 基于当时国内所有专利数据,数据量 2 千万, 并使用 Elasticsearch 实现 SQL-like 形式检索。
    - 使用 TF-IDF 和 PageRank 技术抽取分析专利主要信息。

#### 3. **2017–2019**：北京信息科技大学 **计算机科学硕士**  
- **Yang, Yaofei**, Shuqin Li, Yangsen Zhang, and Hua-Ping Zhang. 2019. “Point the Point: Uyghur Morphological Segmentation Using PointerNetwork with GRU.” In *Chinese Computational Linguistics*, 371–81. Lecture Notes in Computer Science. Cham: Springer International Publishing. https://doi.org/10.1007/978-3-030-32381-3_30.
- **Yang, Yaofei**, Hua-Ping Zhang, Linfang Wu, Xin Liu, and Yangsen Zhang. 2020. “Cached Embedding with Random Selection: Optimization Technique to Improve Training Speed of Character-Aware Embedding.” In *Intelligent Information and Database Systems*, 51–62. Lecture Notes in Computer Science. Cham: Springer International Publishing. https://doi.org/10.1007/978-3-030-41964-6_5.
- Wu, Linfang, Hua-Ping Zhang, **Yaofei Yang**, Xin Liu, and Kai Gao. 2020. “Dynamic Prototype Selection by Fusing Attention Mechanism for Few-Shot Relation Classification.” In *Intelligent Information and Database Systems*, 431–41. Lecture Notes in Computer Science. Cham: Springer International Publishing. https://doi.org/10.1007/978-3-030-41964-6_37. 