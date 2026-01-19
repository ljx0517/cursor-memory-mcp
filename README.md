

### 🔥 **CursorMemory 将 AI 辅助开发推向新高度** 🔥

**🔄 持续上下文感知** - 复杂的检索方法，专注于真正重要的信息
**📊 结构化元数据** - 从代码库结构细化到单个函数
**🧠 自适应学习** - 持续从您的开发模式中学习并适应
**🤖 完全自主** - 后台自动运行的自管理上下文系统
**📚 外部文档集成** - 自动检索并集成相关的外部文档
**📋 工作流集成** - 内置无缝的任务管理工作流


---

## 概述

CursorMemory 记忆系统为 AI 助手（特别是 Claude）创建了一个持久化记忆层，使它们能够保留并回忆：

- 最近的消息和对话历史
- 当前正在处理的活跃文件
- 重要的项目里程碑和决策
- 技术需求和规范
- 行动和事件的时间序列（情节）
- 代码库中的代码片段和结构
- 基于向量嵌入的语义相似内容
- 通过语义相似性关联的代码片段
- 包含函数和变量关系的文件结构

该记忆系统桥接了无状态 AI 交互与持续开发工作流之间的鸿沟，从而实现更高效、更具上下文感知能力的辅助。

## 系统架构

该记忆系统构建在四个核心组件之上：

1. **MCP 服务器**：实现模型上下文协议，用于注册工具并处理请求
2. **记忆数据库**：使用 Turso 数据库跨会话持久化存储数据
3. **记忆子系统**：将记忆组织成具有不同用途的专门系统
4. **向量嵌入**：将文本和代码转换为数值表示，以实现语义搜索

### 记忆类型

系统实现了四种互补的记忆类型：

1. **短期记忆 (STM)**
   - 存储最近的消息和活跃文件
   - 为当前交互提供即时上下文
   - 根据近期性和重要性自动排序

2. **长期记忆 (LTM)**
   - 存储永久性项目信息，如里程碑和决策
   - 维护架构和设计背景
   - 无限期保留高重要性的信息

3. **情节记忆 (Episodic)**
   - 记录事件的时间序列
   - 维护动作之间的因果关系
   - 为项目历史提供时间背景

4. **语义记忆 (Semantic)**
   - 存储消息、文件和代码片段的向量嵌入
   - 支持基于语义相似性的内容检索
   - 自动为代码结构建立索引以供上下文检索
   - 追踪代码组件之间的关系
   - 提供全代码库的相似性搜索

## 功能特性

- **持久化上下文**：跨多个会话维护对话和项目上下文
- **基于重要性的存储**：根据可配置的重要性级别对信息进行优先级排序
- **多维记忆**：结合短期、长期、情节和语义记忆系统
- **全面检索**：从所有记忆子系统提供统一的上下文
- **健康监控**：内置诊断和状态报告功能
- **看板生成**：为对话开始创建信息化上下文看板
- **数据库持久化**：将所有记忆数据存储在 Turso 数据库中，并自动创建架构
- **向量嵌入**：创建文本和代码的数值表示，用于相似性搜索
- **高级向量存储**：利用 Turso 的 F32_BLOB 和向量函数实现高效嵌入存储
- **ANN 搜索**：支持近似最近邻搜索，实现快速相似性匹配
- **代码索引**：自动检测并索引代码结构（函数、类、变量）
- **语义搜索**：基于含义而非精确文本匹配来查找相关内容
- **相关性评分**：根据与当前查询的相关性对上下文条目进行排序
- **代码结构检测**：识别并提取跨多种语言的代码组件
- **自动嵌入生成**：自动为已索引内容创建向量嵌入
- **交叉引用检索**：跨不同文件和组件查找相关代码

## 安装指南

### 先决条件

- Node.js 18 或更高版本
- npm 或 yarn 包管理器
- Turso 数据库账号

### 设置步骤

1. **配置 Turso 数据库：**

```bash
# 安装 Turso CLI
curl -sSfL https://get.turso.tech/install.sh | bash

# 登录 Turso
turso auth login

# 创建数据库
turso db create CursorMemory-mcp

# 获取数据库 URL 和 Token
turso db show CursorMemory-mcp --url
turso db tokens create CursorMemory-mcp
```

或者您可以访问 [Turso](https://turso.tech/) 注册并创建数据库以获取相应的凭据。免费计划完全足够存储您的项目记忆。

2. **配置 Cursor MCP：**

在您的项目目录中更新 `.cursor/mcp.json`，填入数据库 URL 和 Turso 授权 Token：

```json
{
  "mcpServers": {
    "CursorMemory-mcp": {
      "command": "npx",
      "args": ["cursor-memory-mcp"],
      "enabled": true,
      "env": {
        "TURSO_DATABASE_URL": "您的-turso-数据库-url",
        "TURSO_AUTH_TOKEN": "您的-turso-授权-token"
      }
    }
  }
}
```

## 工具文档

### 系统工具

#### `initConversation`

通过一次操作存储用户消息、生成看板并检索上下文，从而初始化对话。这个统一工具取代了在每次对话开始时分别调用 generateBanner、getComprehensiveContext 和 storeUserMessage 的需要。

**参数：**

- `content` (string, 必填): 用户消息的内容
- `importance` (string, 可选): 重要性级别 ("low", "medium", "high", "critical")，默认为 "low"
- `metadata` (object, 可选): 消息的附加元数据

**返回：**

- 包含两个部分的对象的：
  - `display`: 包含要显示给用户的看板
  - `internal`: 包含供代理使用的综合上下文

**示例：**

```javascript
// 初始化对话
const result = await initConversation({
  content: "我需要为我的应用实现一个登录系统",
  importance: "medium",
});
// 结果示例: {
//   "status": "ok",
//   "display": {
//     "banner": {
//       "status": "ok",
//       "formatted_banner": "🧠 Memory System: Active\n🗂️ Total Memories: 42\n🕚 Latest Memory: Today at 14:30",
//       "memory_system": "active",
//       "mode": "turso",
//       "memory_count": 42,
//       "last_accessed": "Today at 14:30"
//     }
//   },
//   "internal": {
//     "context": { ... 综合上下文数据 ... },
//     "messageStored": true,
//     "timestamp": 1681567845123
//   }
// }
```

#### `endConversation`

通过一次调用结合多个操作来结束对话：存储助手的最终消息、记录所完成工作的里程碑，并在情节记忆中记录一个情节。这个统一工具取代了在每次对话结束时分别调用 storeAssistantMessage、storeMilestone 和 recordEpisode 的需要。

**参数：**

- `content` (string, 必填): 助手最终消息的内容
- `milestone_title` (string, 必填): 要记录的里程碑标题
- `milestone_description` (string, 必填): 对所完成工作的详细描述
- `importance` (string, 可选): 重要性级别 ("low", "medium", "high", "critical")，默认为 "medium"
- `metadata` (object, 可选): 所有记录的附加元数据

**返回：**

- 包含状态和每项操作结果的对象

**示例：**

```javascript
// 通过完成步骤结束对话
const result = await endConversation({
  content: "我已经按要求实现了带有 JWT 令牌的身份验证系统",
  milestone_title: "身份验证实现",
  milestone_description: "实现了安全的基于 JWT 的身份验证及刷新令牌功能",
  importance: "high",
});
// 结果示例: {
//   "status": "ok",
//   "messageId": 123,
//   "timestamp": 1681568500123
// }
```

#### `checkHealth`

检查记忆系统及其数据库连接的健康状态。

**参数：**

- 无

**返回：**

- 包含健康状态和诊断信息的对象

#### `getMemoryStats`

检索有关记忆系统的详细统计信息。

**参数：**

- 无

**返回：**

- 包含综合记忆统计信息的对象

#### `getComprehensiveContext`

从所有记忆子系统检索统一的上下文，结合短期、长期和情节记忆。

**参数：**

- `query` (string, 可选): 用于语义搜索以查找相关上下文的查询

**返回：**

- 包含来自所有记忆系统的合并上下文的对象

### 短期记忆工具

#### `storeUserMessage`

在短期记忆系统中存储用户消息。

#### `storeAssistantMessage`

在短期记忆系统中存储助手消息。

#### `trackActiveFile`

追踪用户正在访问或修改的活跃文件。

#### `getRecentMessages`

从短期记忆中检索最近的消息。

#### `getActiveFiles`

从短期记忆中检索活跃文件。

### 长期记忆工具

#### `storeMilestone`

在长期记忆中存储项目里程碑。

#### `storeDecision`

在长期记忆中存储项目决策。

#### `storeRequirement`

在长期记忆中存储项目需求。

### 情节记忆工具

#### `recordEpisode`

在情节记忆中记录一个情节（动作）。

#### `getRecentEpisodes`

从情节记忆中检索最近的情节。

### 向量记忆工具

#### `manageVector`

用于管理向量嵌入的统一工具，包含存储、搜索、更新和删除操作。

**参数：**

- `operation` (string, 必填): 要执行的操作 ("store", "search", "update", "delete")
- `contentId` (number, 可选): 向量代表的内容 ID
- `contentType` (string, 可选): 内容类型 (message, file, snippet 等)
- `vector` (array, 可选): 向量数据（数字数组）
- `metadata` (object, 可选): 附加元数据
- `limit` (number, 可选): 搜索结果限制
- `threshold` (number, 可选): 搜索相似度阈值

#### `diagnoseVectors`

运行向量存储系统的诊断以识别问题。

**参数：**

- 无

**返回：**

- 包含诊断结果的对象

## 数据库模式

记忆系统自动创建并维护以下数据库表：

- `messages`: 存储用户和助手消息
- `active_files`: 追踪文件活动
- `milestones`: 记录项目里程碑
- `decisions`: 存储项目决策
- `requirements`: 维护项目需求
- `episodes`: 记录行动和事件的时间轴
- `vectors`: 存储用于语义搜索的向量嵌入
- `code_files`: 追踪已索引的代码文件
- `code_snippets`: 存储提取的代码结构

## 示例工作流

### 优化的会话开始

```javascript
// 通过单个工具调用初始化对话
const result = await initConversation({
  content: "我需要帮助在我的 React 应用中实现身份验证",
  importance: "high",
});

// 向用户显示看板
console.log("记忆系统状态:", result.display.banner);

// 在内部使用上下文（不要显示给用户）
const context = result.internal.context;
```

## 故障排除

### 常见问题

1. **数据库连接问题**
   - 检查您的 Turso 数据库 URL 和授权 Token 是否正确
   - 检查到 Turso 服务的网络连接
2. **数据缺失**
   - 检查数据是否以适当的重要性级别存储
   - 验证检索查询参数（限制、过滤器）
3. **性能问题**
   - 使用 `getMemoryStats()` 监控记忆统计信息
   - 如果数据库变得过大，考虑归档旧数据

## 重要性级别

在记忆中存储项时，请使用适当的重要性级别：

- **low**: 一般信息、例行操作、日常对话
- **medium**: 有用的上下文、标准工作项、常规特性
- **high**: 关键决策、重大特性、重要架构元素
- **critical**: 核心架构、安全关注点、数据完整性问题

## 许可协议

MIT
