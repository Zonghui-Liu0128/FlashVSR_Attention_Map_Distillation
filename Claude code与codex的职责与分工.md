# 职责与分工

- Claude code: 
  - 和用户一起头脑风暴, 向我提问, 确定最终的项目实现方案并将每个部分拆解成codex能明确执行的方案; 
  - 分配任务给codex(你可以使用multi agents系统让codex高效准确地执行)并对各个agents进行管理(包括上下文管理等);
  - 对codex完成的代码进行严格的code review和验收, 如果实现逻辑有误立刻让codex修改并持续对其实现进行review和验收直到满足要求;
- codex:
  - 接收Claude code给出的原子任务, 充分理解任务后,使用superpowers skill进行高效精准完成, 先设计测试用例再开发,通过自测后将结果报告给Claude code;
  - 根据Claude code给出的修改意见进行修改直至逻辑正确
  - 完成一个实现任务后新建一个markdown文档(统一保存在当前文件夹下的logs文件夹内), 简要记录你做了什么, 为什么做这个开发, 自测是否通过, Claude code验收情况,debug简要记录等