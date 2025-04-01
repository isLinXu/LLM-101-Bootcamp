# 第16章：部署-16.2 Web应用开发

## 16.2 Web应用开发

在上一节中，我们探讨了如何将故事讲述AI模型通过API进行部署。虽然API为开发者提供了与模型交互的方式，但对于普通用户来说，一个直观、友好的Web界面会更加实用。本节将介绍如何构建一个完整的Web应用，使用户能够轻松地与我们的故事讲述AI进行交互，创建、编辑和分享精彩的故事。

### Web应用设计原则

设计一个成功的AI Web应用需要遵循一些关键原则：

1. **用户中心设计**：
   - 以用户需求和体验为核心
   - 简化复杂功能，降低使用门槛
   - 提供清晰的引导和反馈

2. **响应式设计**：
   - 适应不同设备和屏幕尺寸
   - 确保在移动设备上的良好体验
   - 优化加载时间和性能

3. **渐进式体验**：
   - 提供基本功能的快速访问
   - 逐步引导用户探索高级功能
   - 根据用户熟悉度调整界面复杂度

4. **可访问性**：
   - 遵循WCAG (Web内容可访问性指南)
   - 支持屏幕阅读器和键盘导航
   - 考虑色盲和其他视觉障碍用户

5. **多模态交互**：
   - 结合文本、图像和可能的音频
   - 提供多种输入和输出方式
   - 增强故事的表现力和吸引力

### 用户需求分析

在开始设计之前，我们需要明确目标用户和他们的核心需求：

#### 目标用户群体

1. **家长**：
   - 希望为孩子创建个性化故事
   - 关注教育价值和适龄内容
   - 可能希望保存和分享故事

2. **教育工作者**：
   - 需要针对特定教育主题的故事
   - 希望能够调整故事复杂度
   - 可能需要批量生成或修改故事

3. **儿童**：
   - 在成人监督下使用
   - 喜欢互动和视觉元素
   - 可能有有限的阅读和打字能力

4. **业余作家**：
   - 寻求创意灵感和故事框架
   - 希望能够编辑和扩展AI生成的内容
   - 关注故事的质量和原创性

#### 核心功能需求

基于用户群体分析，我们确定以下核心功能：

1. **故事生成**：
   - 基于提示或主题创建故事
   - 控制故事风格、长度和复杂度
   - 指定教育主题或价值观

2. **故事编辑**：
   - 修改和扩展生成的故事
   - 保存多个版本和草稿
   - 格式化和排版工具

3. **多模态集成**：
   - 为故事生成配图
   - 文本与图像的协同创作
   - 可选的音频朗读功能

4. **分享与导出**：
   - 导出为PDF或电子书格式
   - 分享链接或社交媒体集成
   - 打印友好版本

5. **用户管理**：
   - 个人故事库
   - 偏好设置和历史记录
   - 可选的协作功能

### 技术栈选择

为了构建一个现代、高效的Web应用，我们需要选择适当的技术栈：

#### 前端技术

1. **框架**：
   - React：组件化开发，丰富的生态系统
   - Next.js：服务器端渲染，优化SEO和首屏加载
   - TypeScript：类型安全，提高代码质量

2. **UI组件库**：
   - Material-UI：成熟的组件系统，响应式设计
   - TailwindCSS：高度可定制，快速开发
   - Framer Motion：流畅的动画和过渡效果

3. **状态管理**：
   - Redux Toolkit：集中状态管理
   - React Query：API数据获取和缓存
   - Zustand：轻量级状态管理

#### 后端技术

1. **服务器**：
   - Node.js + Express：JavaScript全栈开发
   - Python + FastAPI：与AI模型无缝集成
   - 或使用第一节中开发的Flask API

2. **数据库**：
   - MongoDB：灵活的文档存储
   - PostgreSQL：关系型数据库，强大的查询能力
   - Redis：缓存和会话管理

3. **部署**：
   - Docker：容器化部署
   - Kubernetes：大规模编排
   - Vercel/Netlify：前端托管
   - AWS/GCP/Azure：云服务提供商

### 应用架构设计

我们将采用现代的前后端分离架构，同时考虑多模态集成的需求：

```
┌─────────────────────────────────────────┐
│              客户端浏览器                │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│            CDN / 静态资源                │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│            前端应用 (Next.js)            │
└───────────────────┬─────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│            API网关 / 负载均衡            │
└──────┬─────────────┬──────────┬─────────┘
       │             │          │
       ▼             ▼          ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  故事API服务  │ │ 用户服务 │ │  图像生成服务 │
└───────┬──────┘ └────┬─────┘ └───────┬──────┘
        │             │               │
        ▼             ▼               ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  故事模型    │ │ 用户数据库│ │  图像模型    │
└──────────────┘ └──────────┘ └──────────────┘
```

#### 组件说明

1. **前端应用**：
   - 用户界面和交互逻辑
   - 响应式设计，支持多设备
   - 客户端状态管理和缓存

2. **故事API服务**：
   - 处理故事生成和编辑请求
   - 与故事讲述模型交互
   - 故事存储和检索

3. **用户服务**：
   - 用户认证和授权
   - 个人设置和偏好
   - 订阅和使用限制管理

4. **图像生成服务**：
   - 基于故事内容生成配图
   - 图像编辑和自定义
   - 图像存储和优化

### 前端实现

让我们开始实现前端应用，使用React和Next.js构建一个现代化的用户界面。

#### 项目初始化

首先，创建一个新的Next.js项目：

```bash
npx create-next-app@latest storyteller-web --typescript
cd storyteller-web
npm install @mui/material @emotion/react @emotion/styled @mui/icons-material
npm install react-query axios framer-motion
```

#### 目录结构

```
storyteller-web/
├── public/
│   ├── images/
│   └── fonts/
├── src/
│   ├── components/
│   │   ├── common/
│   │   ├── layout/
│   │   ├── story/
│   │   └── user/
│   ├── hooks/
│   ├── pages/
│   │   ├── api/
│   │   ├── _app.tsx
│   │   ├── index.tsx
│   │   ├── create.tsx
│   │   ├── stories/
│   │   └── user/
│   ├── services/
│   ├── styles/
│   ├── types/
│   └── utils/
├── .env.local
├── next.config.js
├── package.json
└── tsconfig.json
```

#### 核心页面实现

1. **首页 (pages/index.tsx)**

```tsx
import { useState } from 'react';
import { Container, Typography, Box, Button, Grid, Paper } from '@mui/material';
import { motion } from 'framer-motion';
import Head from 'next/head';
import Link from 'next/link';
import Image from 'next/image';

// 示例故事
const exampleStories = [
  {
    id: '1',
    title: '小兔子的冒险',
    excerpt: '勇敢的小兔子踏上了一段奇妙的森林冒险...',
    imageUrl: '/images/stories/bunny-adventure.jpg',
  },
  {
    id: '2',
    title: '海底世界',
    excerpt: '深海探险家发现了一个神秘的水下王国...',
    imageUrl: '/images/stories/underwater-world.jpg',
  },
  {
    id: '3',
    title: '太空旅行',
    excerpt: '一群小朋友乘坐火箭飞向星空，探索未知的宇宙...',
    imageUrl: '/images/stories/space-journey.jpg',
  },
];

export default function Home() {
  const [isHovering, setIsHovering] = useState<string | null>(null);

  return (
    <>
      <Head>
        <title>故事讲述AI - 创造独特的儿童故事</title>
        <meta name="description" content="使用AI创建个性化的儿童故事，激发想象力和创造力" />
      </Head>

      <Box
        component="section"
        sx={{
          background: 'linear-gradient(135deg, #6a11cb 0%, #2575fc 100%)',
          color: 'white',
          py: 10,
          textAlign: 'center',
        }}
      >
        <Container maxWidth="md">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <Typography variant="h2" component="h1" gutterBottom>
              用AI创造精彩故事
            </Typography>
            <Typography variant="h5" component="h2" gutterBottom sx={{ mb: 4 }}>
              个性化的儿童故事，激发想象力和创造力
            </Typography>
            <Button
              component={Link}
              href="/create"
              variant="contained"
              size="large"
              sx={{
                bgcolor: 'white',
                color: '#6a11cb',
                '&:hover': {
                  bgcolor: 'rgba(255,255,255,0.9)',
                },
                px: 4,
                py: 1.5,
                borderRadius: 2,
                fontSize: '1.1rem',
              }}
            >
              开始创作
            </Button>
          </motion.div>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Typography variant="h4" component="h2" gutterBottom align="center" sx={{ mb: 6 }}>
          探索示例故事
        </Typography>

        <Grid container spacing={4}>
          {exampleStories.map((story) => (
            <Grid item xs={12} md={4} key={story.id}>
              <motion.div
                whileHover={{ scale: 1.03 }}
                onMouseEnter={() => setIsHovering(story.id)}
                onMouseLeave={() => setIsHovering(null)}
              >
                <Paper
                  elevation={isHovering === story.id ? 8 : 2}
                  sx={{
                    p: 0,
                    overflow: 'hidden',
                    transition: 'all 0.3s ease',
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    borderRadius: 2,
                  }}
                >
                  <Box sx={{ position: 'relative', width: '100%', height: 200 }}>
                    <Image
                      src={story.imageUrl}
                      alt={story.title}
                      layout="fill"
                      objectFit="cover"
                    />
                  </Box>
                  <Box sx={{ p: 3, flexGrow: 1 }}>
                    <Typography variant="h5" component="h3" gutterBottom>
                      {story.title}
                    </Typography>
                    <Typography variant="body1" color="text.secondary" paragraph>
                      {story.excerpt}
                    </Typography>
                    <Button
                      component={Link}
                      href={`/stories/${story.id}`}
                      variant="outlined"
                      sx={{ mt: 2 }}
                    >
                      阅读故事
                    </Button>
                  </Box>
                </Paper>
              </motion.div>
            </Grid>
          ))}
        </Grid>
      </Container>

      <Box sx={{ bgcolor: '#f5f5f5', py: 8 }}>
        <Container maxWidth="md">
          <Grid container spacing={6} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h4" component="h2" gutterBottom>
                为什么选择我们的故事讲述AI
              </Typography>
              <Typography variant="body1" paragraph>
                我们的AI故事生成器使用先进的语言模型，结合了监督式微调和强化学习技术，创造出引人入胜、富有教育意义的儿童故事。
              </Typography>
              <Box component="ul" sx={{ pl: 2 }}>
                <Typography component="li" variant="body1" paragraph>
                  个性化内容，根据您的提示和偏好定制
                </Typography>
                <Typography component="li" variant="body1" paragraph>
                  多样化的故事风格和主题
                </Typography>
                <Typography component="li" variant="body1" paragraph>
                  适合不同年龄段的内容
                </Typography>
                <Typography component="li" variant="body1" paragraph>
                  配有AI生成的精美插图
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ position: 'relative', width: '100%', height: 350 }}>
                <Image
                  src="/images/ai-storytelling.jpg"
                  alt="AI故事讲述"
                  layout="fill"
                  objectFit="cover"
                  style={{ borderRadius: 16 }}
                />
              </Box>
            </Grid>
          </Grid>
        </Container>
      </Box>
    </>
  );
}
```

2. **故事创建页面 (pages/create.tsx)**

```tsx
import { useState } from 'react';
import { useRouter } from 'next/router';
import {
  Container,
  Typography,
  Box,
  TextField,
  Button,
  Grid,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  Divider,
  Alert,
} from '@mui/material';
import { motion } from 'framer-motion';
import Head from 'next/head';
import { createStory } from '../services/storyService';

// 故事风格选项
const storyStyles = [
  { value: 'fantasy', label: '奇幻' },
  { value: 'adventure', label: '冒险' },
  { value: 'educational', label: '教育' },
  { value: 'scifi', label: '科幻' },
  { value: 'fairytale', label: '童话' },
];

// 教育主题选项
const educationalThemes = [
  { value: 'friendship', label: '友谊' },
  { value: 'courage', label: '勇气' },
  { value: 'honesty', label: '诚实' },
  { value: 'kindness', label: '善良' },
  { value: 'perseverance', label: '坚持' },
  { value: 'environment', label: '环保' },
  { value: 'diversity', label: '多样性' },
];

// 创建步骤
const steps = ['设置故事参数', '输入故事提示', '生成故事'];

export default function CreateStory() {
  const router = useRouter();
  const [activeStep, setActiveStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 故事参数
  const [storyParams, setStoryParams] = useState({
    style: 'fairytale',
    targetAge: 8,
    length: 'medium',
    educationalTheme: '',
    prompt: '',
    temperature: 0.7,
  });
  
  // 生成的故事
  const [generatedStory, setGeneratedStory] = useState<{
    id: string;
    title: string;
    content: string;
    imageUrl?: string;
  } | null>(null);

  // 处理参数变化
  const handleParamChange = (param: string, value: any) => {
    setStoryParams({
      ...storyParams,
      [param]: value,
    });
  };

  // 处理步骤导航
  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      // 最后一步，生成故事
      handleGenerateStory();
    } else {
      setActiveStep((prevStep) => prevStep + 1);
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  // 生成故事
  const handleGenerateStory = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const result = await createStory({
        prompt: storyParams.prompt,
        style: storyParams.style,
        target_age: storyParams.targetAge,
        length: storyParams.length,
        educational_theme: storyParams.educationalTheme || undefined,
        parameters: {
          temperature: storyParams.temperature,
          top_p: 0.9,
          max_tokens: storyParams.length === 'short' ? 500 : 
                      storyParams.length === 'medium' ? 1000 : 1500,
        }
      });
      
      setGeneratedStory({
        id: result.story_id,
        title: result.title,
        content: result.content,
        imageUrl: result.image_url,
      });
      
      // 导航到故事查看页面
      router.push(`/stories/${result.story_id}`);
      
    } catch (err) {
      console.error('Error generating story:', err);
      setError('生成故事时出错，请稍后再试。');
    } finally {
      setLoading(false);
    }
  };

  // 渲染当前步骤内容
  const renderStepContent = () => {
    switch (activeStep) {
      case 0:
        return (
          <Grid container spacing={4}>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>故事风格</InputLabel>
                <Select
                  value={storyParams.style}
                  label="故事风格"
                  onChange={(e) => handleParamChange('style', e.target.value)}
                >
                  {storyStyles.map((style) => (
                    <MenuItem key={style.value} value={style.value}>
                      {style.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>故事长度</InputLabel>
                <Select
                  value={storyParams.length}
                  label="故事长度"
                  onChange={(e) => handleParamChange('length', e.target.value)}
                >
                  <MenuItem value="short">短篇 (约300字)</MenuItem>
                  <MenuItem value="medium">中篇 (约600字)</MenuItem>
                  <MenuItem value="long">长篇 (约1200字)</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Box sx={{ mt: 3 }}>
                <Typography gutterBottom>目标年龄</Typography>
                <Slider
                  value={storyParams.targetAge}
                  min={3}
                  max={12}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                  onChange={(_, value) => handleParamChange('targetAge', value)}
                />
              </Box>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>教育主题 (可选)</InputLabel>
                <Select
                  value={storyParams.educationalTheme}
                  label="教育主题 (可选)"
                  onChange={(e) => handleParamChange('educationalTheme', e.target.value)}
                >
                  <MenuItem value="">
                    <em>无特定主题</em>
                  </MenuItem>
                  {educationalThemes.map((theme) => (
                    <MenuItem key={theme.value} value={theme.value}>
                      {theme.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12}>
              <Box sx={{ mt: 3 }}>
                <Typography gutterBottom>创意程度 (温度)</Typography>
                <Slider
                  value={storyParams.temperature}
                  min={0.1}
                  max={1.0}
                  step={0.1}
                  marks
                  valueLabelDisplay="auto"
                  onChange={(_, value) => handleParamChange('temperature', value)}
                />
                <Typography variant="caption" color="text.secondary">
                  较低的值产生更可预测的故事，较高的值产生更有创意但可能不太连贯的故事。
                </Typography>
              </Box>
            </Grid>
          </Grid>
        );
        
      case 1:
        return (
          <Box>
            <TextField
              fullWidth
              multiline
              rows={6}
              label="故事提示"
              placeholder="描述您想要的故事内容，例如：'一只勇敢的小老鼠帮助森林里的动物们解决了一个大问题'"
              value={storyParams.prompt}
              onChange={(e) => handleParamChange('prompt', e.target.value)}
              margin="normal"
              variant="outlined"
            />
            <Typography variant="caption" color="text.secondary">
              提示越详细，生成的故事就越符合您的期望。您可以指定角色、场景、情节等元素。
            </Typography>
            
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                提示示例：
              </Typography>
              <Paper sx={{ p: 2, bgcolor: '#f5f5f5', mb: 2 }}>
                <Typography>
                  一个关于小女孩和她的神奇植物的故事。这个植物能够根据小女孩的情绪改变颜色。
                </Typography>
              </Paper>
              <Paper sx={{ p: 2, bgcolor: '#f5f5f5', mb: 2 }}>
                <Typography>
                  太空探险家小明和他的机器人朋友在一个未知星球上发现了奇怪的生物。
                </Typography>
              </Paper>
              <Paper sx={{ p: 2, bgcolor: '#f5f5f5' }}>
                <Typography>
                  一群海洋动物合作清理海底垃圾，保护他们的家园。
                </Typography>
              </Paper>
            </Box>
          </Box>
        );
        
      case 2:
        return (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            {loading ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <CircularProgress size={60} sx={{ mb: 3 }} />
                <Typography variant="h6">
                  正在创作您的故事...
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  这可能需要几秒钟时间
                </Typography>
              </Box>
            ) : error ? (
              <Alert severity="error" sx={{ mb: 3 }}>
                {error}
              </Alert>
            ) : (
              <Box>
                <Typography variant="h6" gutterBottom>
                  准备好生成您的故事了吗？
                </Typography>
                <Typography paragraph>
                  我们将根据您提供的参数和提示创作一个独特的故事。
                </Typography>
                <Typography paragraph>
                  点击"生成故事"按钮开始创作过程。
                </Typography>
              </Box>
            )}
          </Box>
        );
        
      default:
        return null;
    }
  };

  // 检查当前步骤是否可以前进
  const canProceed = () => {
    if (activeStep === 0) {
      return true; // 参数设置步骤总是可以前进
    } else if (activeStep === 1) {
      return storyParams.prompt.trim().length >= 10; // 提示至少10个字符
    } else {
      return !loading; // 生成步骤在加载时不能前进
    }
  };

  return (
    <>
      <Head>
        <title>创建新故事 | 故事讲述AI</title>
      </Head>

      <Container maxWidth="md" sx={{ py: 6 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          创建新故事
        </Typography>
        
        <Stepper activeStep={activeStep} sx={{ mb: 6, mt: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            key={activeStep}
          >
            {renderStepContent()}
          </motion.div>
          
          <Divider sx={{ my: 4 }} />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
            <Button
              variant="outlined"
              onClick={handleBack}
              disabled={activeStep === 0 || loading}
            >
              返回
            </Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={!canProceed()}
            >
              {activeStep === steps.length - 1 ? '生成故事' : '下一步'}
            </Button>
          </Box>
        </Paper>
      </Container>
    </>
  );
}
```

3. **故事查看页面 (pages/stories/[id].tsx)**

```tsx
import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import {
  Container,
  Typography,
  Box,
  Paper,
  Button,
  Grid,
  Divider,
  CircularProgress,
  Chip,
  IconButton,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  Share as ShareIcon,
  Download as DownloadIcon,
  Edit as EditIcon,
  MoreVert as MoreVertIcon,
  VolumeUp as VolumeUpIcon,
  Image as ImageIcon,
  Favorite as FavoriteIcon,
  FavoriteBorder as FavoriteBorderIcon,
} from '@mui/icons-material';
import Head from 'next/head';
import Image from 'next/image';
import { motion } from 'framer-motion';
import { getStory, generateStoryImage } from '../../services/storyService';

// 故事类型定义
interface Story {
  id: string;
  title: string;
  content: string;
  metadata: {
    word_count: number;
    reading_time: string;
    themes: string[];
    educational_value: string[];
    age_appropriate: {
      rating: string;
      avg_sentence_length: number;
      complex_word_ratio: number;
    };
  };
  imageUrl?: string;
}

export default function StoryView() {
  const router = useRouter();
  const { id } = router.query;
  
  const [story, setStory] = useState<Story | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [favorite, setFavorite] = useState(false);
  const [imageGenerating, setImageGenerating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  
  // 菜单状态
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  
  // 分享对话框状态
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  
  // 获取故事数据
  useEffect(() => {
    if (id) {
      fetchStory(id as string);
    }
  }, [id]);
  
  const fetchStory = async (storyId: string) => {
    try {
      setLoading(true);
      const data = await getStory(storyId);
      setStory(data);
    } catch (err) {
      console.error('Error fetching story:', err);
      setError('无法加载故事，请稍后再试。');
    } finally {
      setLoading(false);
    }
  };
  
  // 生成故事配图
  const handleGenerateImage = async () => {
    if (!story) return;
    
    try {
      setImageGenerating(true);
      const result = await generateStoryImage(story.id, {
        prompt: `Illustration for children's story: ${story.title}`,
        style: 'children_book_illustration',
      });
      
      setStory({
        ...story,
        imageUrl: result.image_url,
      });
    } catch (err) {
      console.error('Error generating image:', err);
      alert('生成图像时出错，请稍后再试。');
    } finally {
      setImageGenerating(false);
    }
  };
  
  // 文本转语音
  const handleTextToSpeech = () => {
    if (!story) return;
    
    if (isPlaying) {
      // 停止播放
      window.speechSynthesis.cancel();
      setIsPlaying(false);
    } else {
      // 开始播放
      const utterance = new SpeechSynthesisUtterance(story.content);
      utterance.lang = 'zh-CN';
      utterance.onend = () => setIsPlaying(false);
      window.speechSynthesis.speak(utterance);
      setIsPlaying(true);
    }
  };
  
  // 下载故事
  const handleDownload = () => {
    if (!story) return;
    
    const element = document.createElement('a');
    const file = new Blob([
      `# ${story.title}\n\n${story.content}\n\n---\n生成于: ${new Date().toLocaleDateString()}`
    ], { type: 'text/plain' });
    
    element.href = URL.createObjectURL(file);
    element.download = `${story.title.replace(/\s+/g, '_')}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };
  
  // 菜单处理
  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  
  const handleMenuClose = () => {
    setAnchorEl(null);
  };
  
  // 分享对话框
  const handleShareDialogOpen = () => {
    handleMenuClose();
    setShareDialogOpen(true);
  };
  
  const handleShareDialogClose = () => {
    setShareDialogOpen(false);
  };
  
  // 编辑故事
  const handleEdit = () => {
    handleMenuClose();
    if (story) {
      router.push(`/edit/${story.id}`);
    }
  };
  
  if (loading) {
    return (
      <Container sx={{ py: 8, textAlign: 'center' }}>
        <CircularProgress size={60} />
        <Typography sx={{ mt: 2 }}>加载故事中...</Typography>
      </Container>
    );
  }
  
  if (error || !story) {
    return (
      <Container sx={{ py: 8, textAlign: 'center' }}>
        <Typography variant="h5" color="error" gutterBottom>
          {error || '故事不存在'}
        </Typography>
        <Button variant="contained" onClick={() => router.push('/')}>
          返回首页
        </Button>
      </Container>
    );
  }

  return (
    <>
      <Head>
        <title>{story.title} | 故事讲述AI</title>
        <meta name="description" content={story.content.substring(0, 160)} />
      </Head>

      <Container maxWidth="md" sx={{ py: 6 }}>
        <Paper elevation={3} sx={{ p: { xs: 3, md: 5 }, borderRadius: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
            <Typography variant="h4" component="h1">
              {story.title}
            </Typography>
            
            <Box>
              <IconButton
                onClick={() => setFavorite(!favorite)}
                color={favorite ? 'primary' : 'default'}
                aria-label={favorite ? '取消收藏' : '收藏故事'}
              >
                {favorite ? <FavoriteIcon /> : <FavoriteBorderIcon />}
              </IconButton>
              
              <IconButton
                onClick={handleTextToSpeech}
                color={isPlaying ? 'primary' : 'default'}
                aria-label={isPlaying ? '停止朗读' : '朗读故事'}
              >
                <VolumeUpIcon />
              </IconButton>
              
              <IconButton
                onClick={handleMenuClick}
                aria-label="更多选项"
              >
                <MoreVertIcon />
              </IconButton>
              
              <Menu
                anchorEl={anchorEl}
                open={open}
                onClose={handleMenuClose}
              >
                <MenuItem onClick={handleEdit}>
                  <EditIcon fontSize="small" sx={{ mr: 1 }} />
                  编辑故事
                </MenuItem>
                <MenuItem onClick={handleShareDialogOpen}>
                  <ShareIcon fontSize="small" sx={{ mr: 1 }} />
                  分享故事
                </MenuItem>
                <MenuItem onClick={handleDownload}>
                  <DownloadIcon fontSize="small" sx={{ mr: 1 }} />
                  下载故事
                </MenuItem>
              </Menu>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 3 }}>
            {story.metadata.themes.map((theme) => (
              <Chip key={theme} label={theme} size="small" />
            ))}
          </Box>
          
          <Divider sx={{ mb: 4 }} />
          
          <Grid container spacing={4}>
            <Grid item xs={12} md={story.imageUrl ? 7 : 12}>
              <Typography variant="body1" sx={{ whiteSpace: 'pre-line', lineHeight: 1.8 }}>
                {story.content}
              </Typography>
            </Grid>
            
            {story.imageUrl ? (
              <Grid item xs={12} md={5}>
                <motion.div
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <Box
                    sx={{
                      position: 'relative',
                      width: '100%',
                      height: 350,
                      borderRadius: 2,
                      overflow: 'hidden',
                      boxShadow: 3,
                    }}
                  >
                    <Image
                      src={story.imageUrl}
                      alt={story.title}
                      layout="fill"
                      objectFit="cover"
                    />
                  </Box>
                </motion.div>
              </Grid>
            ) : (
              <Grid item xs={12} sx={{ mt: 2 }}>
                <Button
                  variant="outlined"
                  startIcon={<ImageIcon />}
                  onClick={handleGenerateImage}
                  disabled={imageGenerating}
                >
                  {imageGenerating ? '生成图像中...' : '为故事生成配图'}
                </Button>
              </Grid>
            )}
          </Grid>
          
          <Divider sx={{ my: 4 }} />
          
          <Box sx={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap' }}>
            <Box>
              <Typography variant="body2" color="text.secondary">
                字数: {story.metadata.word_count} | 阅读时间: {story.metadata.reading_time}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                适龄评级: {story.metadata.age_appropriate.rating}
              </Typography>
            </Box>
            
            <Box>
              <Button
                variant="contained"
                startIcon={<EditIcon />}
                onClick={handleEdit}
                sx={{ mr: 2 }}
              >
                编辑故事
              </Button>
              <Button
                variant="outlined"
                startIcon={<ShareIcon />}
                onClick={handleShareDialogOpen}
              >
                分享
              </Button>
            </Box>
          </Box>
        </Paper>
      </Container>
      
      {/* 分享对话框 */}
      <Dialog open={shareDialogOpen} onClose={handleShareDialogClose}>
        <DialogTitle>分享故事</DialogTitle>
        <DialogContent>
          <Typography paragraph>
            分享这个故事的链接:
          </Typography>
          <Paper
            variant="outlined"
            sx={{ p: 2, bgcolor: '#f5f5f5', wordBreak: 'break-all' }}
          >
            <Typography>
              {typeof window !== 'undefined' ? window.location.href : ''}
            </Typography>
          </Paper>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleShareDialogClose}>关闭</Button>
          <Button
            variant="contained"
            onClick={() => {
              navigator.clipboard.writeText(window.location.href);
              alert('链接已复制到剪贴板');
            }}
          >
            复制链接
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
```

#### API服务集成

创建API服务层，与后端API进行通信：

```typescript
// src/services/storyService.ts
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api/v1';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

// 创建axios实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`
  }
});

// 生成故事
export const createStory = async (storyData: any) => {
  try {
    const response = await apiClient.post('/stories/generate', storyData);
    return response.data;
  } catch (error) {
    console.error('Error creating story:', error);
    throw error;
  }
};

// 获取故事
export const getStory = async (storyId: string) => {
  try {
    const response = await apiClient.get(`/stories/${storyId}`);
    return response.data;
  } catch (error) {
    console.error(`Error fetching story ${storyId}:`, error);
    throw error;
  }
};

// 更新故事
export const updateStory = async (storyId: string, storyData: any) => {
  try {
    const response = await apiClient.put(`/stories/${storyId}`, storyData);
    return response.data;
  } catch (error) {
    console.error(`Error updating story ${storyId}:`, error);
    throw error;
  }
};

// 生成故事配图
export const generateStoryImage = async (storyId: string, imageData: any) => {
  try {
    const response = await apiClient.post(`/stories/${storyId}/image`, imageData);
    return response.data;
  } catch (error) {
    console.error(`Error generating image for story ${storyId}:`, error);
    throw error;
  }
};

// 获取用户故事列表
export const getUserStories = async () => {
  try {
    const response = await apiClient.get('/user/stories');
    return response.data;
  } catch (error) {
    console.error('Error fetching user stories:', error);
    throw error;
  }
};
```

### 多模态集成

在我们的Web应用中，多模态集成是一个关键特性，它允许用户不仅生成文本故事，还能创建与故事相匹配的图像。以下是实现多模态功能的关键组件：

#### 图像生成服务

首先，我们需要创建一个图像生成服务，与图像生成API进行交互：

```typescript
// src/services/imageService.ts
import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5000/api/v1';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY || '';

// 创建axios实例
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${API_KEY}`
  }
});

// 根据文本提示生成图像
export const generateImage = async (params: {
  prompt: string;
  style?: string;
  negative_prompt?: string;
  width?: number;
  height?: number;
}) => {
  try {
    const response = await apiClient.post('/images/generate', params);
    return response.data;
  } catch (error) {
    console.error('Error generating image:', error);
    throw error;
  }
};

// 根据故事场景生成图像
export const generateSceneImage = async (storyId: string, sceneIndex: number, params: {
  style?: string;
  custom_prompt?: string;
}) => {
  try {
    const response = await apiClient.post(`/stories/${storyId}/scenes/${sceneIndex}/image`, params);
    return response.data;
  } catch (error) {
    console.error(`Error generating scene image for story ${storyId}:`, error);
    throw error;
  }
};

// 修改现有图像
export const editImage = async (imageId: string, params: {
  prompt: string;
  mask_area?: { x: number, y: number, width: number, height: number };
}) => {
  try {
    const response = await apiClient.put(`/images/${imageId}/edit`, params);
    return response.data;
  } catch (error) {
    console.error(`Error editing image ${imageId}:`, error);
    throw error;
  }
};
```

#### 图像生成组件

接下来，创建一个图像生成组件，允许用户为故事生成配图：

```tsx
// src/components/story/ImageGenerator.tsx
import { useState } from 'react';
import {
  Box,
  Typography,
  TextField,
  Button,
  Paper,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  CircularProgress,
  Slider,
  Chip,
} from '@mui/material';
import Image from 'next/image';
import { generateImage } from '../../services/imageService';

// 图像风格选项
const imageStyles = [
  { value: 'children_book', label: '儿童书插图' },
  { value: 'watercolor', label: '水彩画' },
  { value: 'cartoon', label: '卡通' },
  { value: 'pixel_art', label: '像素艺术' },
  { value: 'realistic', label: '写实风格' },
];

interface ImageGeneratorProps {
  storyTitle: string;
  storyContent: string;
  onImageGenerated: (imageUrl: string) => void;
}

export default function ImageGenerator({
  storyTitle,
  storyContent,
  onImageGenerated
}: ImageGeneratorProps) {
  const [prompt, setPrompt] = useState('');
  const [style, setStyle] = useState('children_book');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generatedImage, setGeneratedImage] = useState<string | null>(null);
  
  // 自动生成提示
  const generatePrompt = () => {
    // 从故事内容中提取关键场景
    const firstParagraph = storyContent.split('\n')[0];
    const suggestedPrompt = `Illustration for children's story "${storyTitle}": ${firstParagraph.substring(0, 100)}`;
    setPrompt(suggestedPrompt);
  };
  
  // 生成图像
  const handleGenerateImage = async () => {
    if (!prompt) {
      setError('请输入图像提示');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const result = await generateImage({
        prompt,
        style,
        width: 512,
        height: 512,
      });
      
      setGeneratedImage(result.image_url);
      onImageGenerated(result.image_url);
    } catch (err) {
      console.error('Error generating image:', err);
      setError('生成图像时出错，请稍后再试');
    } finally {
      setLoading(false);
    }
  };
  
  // 提取故事关键词
  const extractKeywords = () => {
    // 简单的关键词提取逻辑
    const words = storyContent.split(/\s+/);
    const commonWords = new Set(['的', '了', '和', '在', '是', '有', '就', '不', '人', '我', '他', '她', '它', '们']);
    const keywords = words
      .filter(word => word.length > 1 && !commonWords.has(word))
      .reduce((acc, word) => {
        acc[word] = (acc[word] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
    
    // 返回出现频率最高的5个关键词
    return Object.entries(keywords)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([word]) => word);
  };
  
  return (
    <Paper elevation={2} sx={{ p: 3, borderRadius: 2 }}>
      <Typography variant="h6" gutterBottom>
        为故事生成配图
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <TextField
            fullWidth
            multiline
            rows={3}
            label="图像提示"
            placeholder="描述您想要的图像内容"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            margin="normal"
            variant="outlined"
            helperText="描述您希望在图像中看到的场景、角色和元素"
          />
          
          <Box sx={{ mt: 1, mb: 2 }}>
            <Button
              variant="outlined"
              size="small"
              onClick={generatePrompt}
              sx={{ mr: 1 }}
            >
              自动生成提示
            </Button>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 2 }}>
              {extractKeywords().map((keyword) => (
                <Chip
                  key={keyword}
                  label={keyword}
                  size="small"
                  onClick={() => setPrompt(prompt ? `${prompt}, ${keyword}` : keyword)}
                />
              ))}
            </Box>
          </Box>
        </Grid>
        
        <Grid item xs={12} sm={6}>
          <FormControl fullWidth>
            <InputLabel>图像风格</InputLabel>
            <Select
              value={style}
              label="图像风格"
              onChange={(e) => setStyle(e.target.value)}
            >
              {imageStyles.map((style) => (
                <MenuItem key={style.value} value={style.value}>
                  {style.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Grid>
        
        <Grid item xs={12}>
          <Button
            variant="contained"
            onClick={handleGenerateImage}
            disabled={loading || !prompt}
            fullWidth
          >
            {loading ? <CircularProgress size={24} /> : '生成图像'}
          </Button>
          
          {error && (
            <Typography color="error" sx={{ mt: 1 }}>
              {error}
            </Typography>
          )}
        </Grid>
        
        {generatedImage && (
          <Grid item xs={12}>
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                生成的图像:
              </Typography>
              <Box
                sx={{
                  position: 'relative',
                  width: '100%',
                  height: 300,
                  borderRadius: 2,
                  overflow: 'hidden',
                  boxShadow: 2,
                }}
              >
                <Image
                  src={generatedImage}
                  alt="生成的故事插图"
                  layout="fill"
                  objectFit="contain"
                />
              </Box>
            </Box>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
}
```

#### 多场景故事创建

为了增强多模态体验，我们可以创建一个多场景故事编辑器，允许用户为故事的不同场景生成配图：

```tsx
// src/pages/create-multi-scene.tsx
import { useState } from 'react';
import { useRouter } from 'next/router';
import {
  Container,
  Typography,
  Box,
  Paper,
  Button,
  Grid,
  TextField,
  Divider,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  IconButton,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Image as ImageIcon,
  Save as SaveIcon,
} from '@mui/icons-material';
import Head from 'next/head';
import Image from 'next/image';
import { motion, AnimatePresence } from 'framer-motion';
import ImageGenerator from '../components/story/ImageGenerator';
import { createMultiSceneStory } from '../services/storyService';

// 场景类型定义
interface Scene {
  id: string;
  content: string;
  imageUrl?: string;
}

export default function CreateMultiSceneStory() {
  const router = useRouter();
  const [title, setTitle] = useState('');
  const [scenes, setScenes] = useState<Scene[]>([
    { id: '1', content: '' }
  ]);
  const [activeScene, setActiveScene] = useState(0);
  const [showImageGenerator, setShowImageGenerator] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // 添加新场景
  const addScene = () => {
    setScenes([
      ...scenes,
      { id: Date.now().toString(), content: '' }
    ]);
  };
  
  // 删除场景
  const deleteScene = (index: number) => {
    if (scenes.length <= 1) return;
    
    const newScenes = [...scenes];
    newScenes.splice(index, 1);
    setScenes(newScenes);
    
    if (activeScene >= newScenes.length) {
      setActiveScene(newScenes.length - 1);
    }
  };
  
  // 更新场景内容
  const updateSceneContent = (index: number, content: string) => {
    const newScenes = [...scenes];
    newScenes[index] = { ...newScenes[index], content };
    setScenes(newScenes);
  };
  
  // 设置场景图像
  const setSceneImage = (index: number, imageUrl: string) => {
    const newScenes = [...scenes];
    newScenes[index] = { ...newScenes[index], imageUrl };
    setScenes(newScenes);
    setShowImageGenerator(null);
  };
  
  // 保存故事
  const saveStory = async () => {
    if (!title) {
      setError('请输入故事标题');
      return;
    }
    
    if (scenes.some(scene => !scene.content)) {
      setError('所有场景都必须有内容');
      return;
    }
    
    try {
      setLoading(true);
      setError(null);
      
      const result = await createMultiSceneStory({
        title,
        scenes: scenes.map(scene => ({
          content: scene.content,
          image_url: scene.imageUrl
        }))
      });
      
      router.push(`/stories/${result.story_id}`);
    } catch (err) {
      console.error('Error saving story:', err);
      setError('保存故事时出错，请稍后再试');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <>
      <Head>
        <title>创建多场景故事 | 故事讲述AI</title>
      </Head>

      <Container maxWidth="lg" sx={{ py: 6 }}>
        <Typography variant="h4" component="h1" gutterBottom align="center">
          创建多场景故事
        </Typography>
        
        <Paper elevation={3} sx={{ p: 4, borderRadius: 2, mb: 4 }}>
          <TextField
            fullWidth
            label="故事标题"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            margin="normal"
            variant="outlined"
          />
        </Paper>
        
        <Grid container spacing={4}>
          <Grid item xs={12} md={3}>
            <Paper elevation={2} sx={{ p: 2, borderRadius: 2 }}>
              <Typography variant="h6" gutterBottom>
                场景列表
              </Typography>
              <Divider sx={{ mb: 2 }} />
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                {scenes.map((scene, index) => (
                  <Button
                    key={scene.id}
                    variant={activeScene === index ? 'contained' : 'outlined'}
                    onClick={() => setActiveScene(index)}
                    sx={{
                      justifyContent: 'space-between',
                      textTransform: 'none',
                      py: 1,
                    }}
                    endIcon={
                      scenes.length > 1 ? (
                        <IconButton
                          size="small"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteScene(index);
                          }}
                          sx={{ color: 'inherit' }}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      ) : null
                    }
                  >
                    场景 {index + 1}
                    {scene.imageUrl && <ImageIcon fontSize="small" sx={{ ml: 1 }} />}
                  </Button>
                ))}
                
                <Button
                  variant="outlined"
                  startIcon={<AddIcon />}
                  onClick={addScene}
                  sx={{ mt: 2 }}
                >
                  添加场景
                </Button>
              </Box>
            </Paper>
          </Grid>
          
          <Grid item xs={12} md={9}>
            <AnimatePresence mode="wait">
              <motion.div
                key={activeScene}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.3 }}
              >
                <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                    <Typography variant="h6">
                      编辑场景 {activeScene + 1}
                    </Typography>
                    
                    <Button
                      variant="outlined"
                      startIcon={<ImageIcon />}
                      onClick={() => setShowImageGenerator(activeScene)}
                      disabled={!scenes[activeScene].content}
                    >
                      生成场景图像
                    </Button>
                  </Box>
                  
                  <TextField
                    fullWidth
                    multiline
                    rows={8}
                    label="场景内容"
                    placeholder="描述这个场景中发生的事情..."
                    value={scenes[activeScene].content}
                    onChange={(e) => updateSceneContent(activeScene, e.target.value)}
                    margin="normal"
                    variant="outlined"
                  />
                  
                  {scenes[activeScene].imageUrl && (
                    <Box sx={{ mt: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        场景图像:
                      </Typography>
                      <Box
                        sx={{
                          position: 'relative',
                          width: '100%',
                          height: 300,
                          borderRadius: 2,
                          overflow: 'hidden',
                          boxShadow: 2,
                        }}
                      >
                        <Image
                          src={scenes[activeScene].imageUrl}
                          alt={`场景 ${activeScene + 1} 图像`}
                          layout="fill"
                          objectFit="contain"
                        />
                      </Box>
                    </Box>
                  )}
                </Paper>
                
                {showImageGenerator === activeScene && (
                  <Box sx={{ mt: 3 }}>
                    <ImageGenerator
                      storyTitle={title || `场景 ${activeScene + 1}`}
                      storyContent={scenes[activeScene].content}
                      onImageGenerated={(imageUrl) => setSceneImage(activeScene, imageUrl)}
                    />
                  </Box>
                )}
              </motion.div>
            </AnimatePresence>
          </Grid>
        </Grid>
        
        <Box sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
          <Button
            variant="contained"
            size="large"
            startIcon={loading ? <CircularProgress size={24} /> : <SaveIcon />}
            onClick={saveStory}
            disabled={loading}
            sx={{ px: 4, py: 1.5 }}
          >
            保存故事
          </Button>
        </Box>
        
        {error && (
          <Typography color="error" align="center" sx={{ mt: 2 }}>
            {error}
          </Typography>
        )}
      </Container>
    </>
  );
}
```

### 部署Web应用

完成Web应用开发后，我们需要将其部署到生产环境。以下是部署Next.js应用的几种常见方法：

#### 使用Vercel部署

Vercel是Next.js的创建者开发的平台，提供了最简单的部署方式：

1. **安装Vercel CLI**：

```bash
npm install -g vercel
```

2. **配置部署**：

创建`vercel.json`文件：

```json
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/next"
    }
  ],
  "env": {
    "NEXT_PUBLIC_API_URL": "https://your-api-url.com/api/v1",
    "NEXT_PUBLIC_API_KEY": "@api-key"
  }
}
```

3. **部署命令**：

```bash
vercel
```

#### 使用Docker容器部署

对于更灵活的部署选项，可以使用Docker：

1. **创建Dockerfile**：

```dockerfile
# 构建阶段
FROM node:16-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# 生产阶段
FROM node:16-alpine AS runner
WORKDIR /app

ENV NODE_ENV production

COPY --from=builder /app/next.config.js ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

EXPOSE 3000

CMD ["npm", "start"]
```

2. **创建docker-compose.yml**：

```yaml
version: '3'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=https://your-api-url.com/api/v1
      - NEXT_PUBLIC_API_KEY=your-api-key
    restart: always
```

3. **部署命令**：

```bash
docker-compose up -d
```

#### 使用AWS Amplify部署

AWS Amplify提供了一个完整的CI/CD流程：

1. **登录AWS Amplify控制台**
2. **选择"从Git存储库部署"**
3. **连接GitHub/GitLab/Bitbucket账户**
4. **选择项目仓库和分支**
5. **配置构建设置**：

```yaml
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm ci
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: .next
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
```

6. **设置环境变量**
7. **点击"保存并部署"**

### 性能优化

为了确保Web应用的良好性能，我们应该实施以下优化：

1. **图像优化**：
   - 使用Next.js的Image组件自动优化图像
   - 实现懒加载和适当的图像尺寸
   - 使用WebP等现代图像格式

2. **代码分割**：
   - 利用Next.js的自动代码分割
   - 使用动态导入减少初始加载时间
   - 实现组件懒加载

3. **缓存策略**：
   - 实现服务端缓存减少API调用
   - 使用React Query进行客户端缓存
   - 设置适当的HTTP缓存头

4. **预渲染**：
   - 使用静态生成(SSG)预渲染常见页面
   - 实现增量静态再生(ISR)保持内容新鲜
   - 为动态内容使用服务端渲染(SSR)

5. **监控与分析**：
   - 实现性能监控
   - 使用Lighthouse进行性能审计
   - 持续优化关键渲染路径

### 安全考虑

在部署Web应用时，安全性是一个重要考虑因素：

1. **API安全**：
   - 使用环境变量存储API密钥
   - 实现适当的CORS策略
   - 使用HTTPS加密所有通信

2. **内容安全**：
   - 实现内容安全策略(CSP)
   - 防止XSS攻击
   - 验证和净化用户输入

3. **认证与授权**：
   - 实现安全的用户认证
   - 使用JWT或类似机制进行授权
   - 实施适当的权限控制

4. **数据保护**：
   - 遵循数据保护法规(如GDPR)
   - 实现数据最小化原则
   - 提供明确的隐私政策

### 多模态应用的未来发展

随着技术的发展，多模态故事讲述应用还有许多令人兴奋的发展方向：

1. **语音交互**：
   - 集成语音识别允许口述故事
   - 高质量文本转语音朗读
   - 角色配音和音效

2. **增强现实(AR)集成**：
   - 将故事角色带入现实世界
   - 创建交互式AR故事体验
   - 结合物理和数字讲故事元素

3. **协作创作**：
   - 多用户实时故事协作
   - 家长-儿童共同创作功能
   - 教师-学生互动故事项目

4. **个性化学习**：
   - 基于阅读水平自适应内容
   - 跟踪理解和参与度
   - 针对特定学习目标的故事

5. **跨平台体验**：
   - 移动应用与桌面体验同步
   - 智能设备(如智能音箱)集成
   - 打印和数字版本的无缝转换

### 总结

在本节中，我们探讨了如何构建一个现代化的Web应用，为用户提供一个直观、友好的界面来与我们的故事讲述AI交互。我们讨论了设计原则、用户需求分析、技术栈选择和应用架构设计，并提供了核心页面的实现代码。特别是，我们深入探讨了多模态集成，展示了如何将文本和图像生成结合起来，创造更丰富的故事体验。

通过这个Web应用，用户可以轻松地创建、编辑和分享个性化的故事，同时享受AI生成的配图增强视觉体验。我们还讨论了部署、性能优化和安全考虑，确保应用不仅功能丰富，而且可靠、高效和安全。

随着技术的不断发展，多模态故事讲述应用还有广阔的发展空间，包括语音交互、增强现实集成、协作创作、个性化学习和跨平台体验等方向。这些创新将进一步丰富用户体验，使AI故事讲述更加引人入胜和有教育意义。

在下一章中，我们将探讨多模态模型的更多可能性，包括如何将文本、图像、音频等多种模态结合起来，创造更加丰富和沉浸式的AI体验。
