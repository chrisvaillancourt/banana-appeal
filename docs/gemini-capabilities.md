# Gemini Image Generation Capabilities

This document describes the image generation capabilities available through the Gemini API and how they're exposed (or could be exposed) through Banana Appeal.

## Available Models

| Model | Best For | Features |
|-------|----------|----------|
| `gemini-2.5-flash-image` | Fast generation, simple edits | Quick turnaround, good for iteration |
| `gemini-3-pro-image-preview` | Complex tasks, high quality | 4K output, advanced text rendering, Google Search grounding, multi-image references |

**Current default:** `gemini-2.0-flash-preview-image-generation`

## Core Capabilities

### Text-to-Image Generation

Generate images from text prompts with configurable:

- **Aspect ratios:** 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
- **Resolutions:** 1K (default), 2K, 4K (must be uppercase)
- **Output formats:** PNG, JPEG, WebP

### Image Editing

Modify existing images using natural language instructions:

- Add, remove, or modify elements
- Change colors, lighting, style
- Multi-turn conversational editing (with thought signatures)

### Multi-Image Blending

Combine multiple reference images (up to 14) with a creative prompt:

- Style transfer from reference images
- Character consistency across generations
- Object + human reference images (6 object + 5 human max for Gemini 3 Pro)

### Text Rendering

Gemini 3 Pro Image excels at generating legible, accurately-placed text:

- Logos and branding
- Diagrams with labels
- Infographics
- Posters and marketing materials

### Google Search Grounding (Gemini 3 Pro only)

Generate images with real-time, fact-verified information:

- Current weather visualizations
- Real product mockups
- Up-to-date data visualizations

## Prompting Strategies

### Core Principle

> "Describe the scene, don't just list keywords."

Write narrative prompts that paint a picture rather than comma-separated attributes.

### 1. Photorealistic Scenes

Use photography terminology:

```
A software developer's desk at golden hour, shot with an 85mm portrait lens
at f/1.8. Shallow depth of field with soft bokeh. Three monitors showing
code, mechanical keyboard, and a steaming coffee cup. Warm, natural lighting
from a nearby window.
```

### 2. Technical Diagrams

Be explicit about structure and style:

```
A clean, minimal system architecture diagram showing three microservices
connected to a central API gateway. Each service is a rounded rectangle
with a bold label. Arrows show data flow direction. Use a modern tech
company color palette with blues and teals. White background, no 3D effects.
```

### 3. Infographics

Specify layout and data representation:

```
A vertical infographic comparing REST vs GraphQL APIs. Two columns side by side.
Each row shows a feature comparison with simple icons. Clean sans-serif typography.
Minimal color palette: blue for REST, purple for GraphQL. Include a summary
verdict at the bottom.
```

### 4. UI/UX Mockups

Use design terminology:

```
A wireframe mockup of a developer dashboard. Sidebar navigation on the left
with icons for Home, Projects, Settings. Header with search bar and user avatar.
Main content area shows a grid of project cards with status badges.
Grayscale with subtle shadows. Figma-style presentation.
```

### 5. Text in Images

Be explicit about text content and styling:

```
A GitHub repository social preview card. Bold white text reading "Banana Appeal"
centered on a gradient background (yellow to orange). Below in smaller text:
"AI Image Generation for Claude Code". Clean, modern sans-serif font.
1280x640 pixels aspect ratio.
```

### 6. Algorithm Visualizations

Describe the data structure clearly:

```
A binary search tree visualization with 7 nodes. Root node contains 50.
Left subtree: 25, 12, 37. Right subtree: 75, 62, 87. Each node is a circle
with the number inside. Edges are clean lines with subtle arrows.
Educational style with a light blue background.
```

## Limitations

### Content Restrictions

- No deceptive, harassing, or harmful content
- Subject to Google's Prohibited Use Policy
- All generated images include SynthID watermarks

### Technical Limitations

| Limitation | Details |
|------------|---------|
| Text accuracy | Gemini 2.5 Flash may render text imperfectly; use Gemini 3 Pro for reliable text |
| Character consistency | Requires reference images for consistent human likenesses |
| Precise layouts | AI may not perfectly match exact specifications; best for conceptual work |
| Processing time | 4K and complex prompts take longer than 1K simple generations |

### What Works Best vs. What Doesn't

**Works Well:**
- Conceptual diagrams and illustrations
- Mood boards and style exploration
- Quick UI/UX concept sketches
- Educational visualizations
- Creative compositions

**Use Traditional Tools Instead:**
- Precise technical diagrams (use Mermaid, PlantUML)
- Exact database schemas (use dbdiagram.io)
- Flowcharts with specific text (use Mermaid)
- Pixel-perfect UI designs (use Figma)

## Coding Workflow Use Cases

### Documentation Assets

```
# README hero image
A friendly cartoon banana character wearing sunglasses and holding a paintbrush,
standing next to an easel with a generated landscape. Flat vector style with
bold outlines. Cheerful yellow and orange color scheme.
```

### Architecture Diagrams

```
# Microservices overview
A horizontal architecture diagram showing: User -> Load Balancer -> API Gateway ->
three backend services (Auth, Users, Orders) -> shared PostgreSQL database.
Clean boxes with rounded corners, directional arrows, modern tech aesthetic.
```

### Concept Exploration

```
# Feature mockup
A mobile app screen showing a code review interface. Dark mode theme.
Diff view with green additions and red deletions. Comment bubbles on the side.
Bottom navigation with icons for Files, Changes, Comments, Approve.
```

### Presentations

```
# Slide background
Abstract geometric pattern with subtle code-like elements. Deep navy blue
background with lighter blue accent shapes. Minimalist, professional.
Leaves space for overlaid text on the right side.
```

## API Parameters Reference

### GenerateContentConfig

```python
from google.genai.types import GenerateContentConfig, ImageConfig

config = GenerateContentConfig(
    response_modalities=["IMAGE", "TEXT"],  # Can return both
    image_config=ImageConfig(
        aspect_ratio="16:9",  # See supported ratios above
        image_size="2K",      # 1K, 2K, or 4K (uppercase required)
    ),
    tools=[{"google_search": {}}],  # Optional: enable search grounding
)
```

### Supported Aspect Ratios

| Ratio | Use Case |
|-------|----------|
| 1:1 | Social media posts, avatars |
| 16:9 | Presentations, video thumbnails |
| 9:16 | Mobile screens, stories |
| 4:3 | Traditional displays |
| 3:2 | Photography standard |
| 21:9 | Ultrawide, cinematic |

## Future Enhancements

Features from the Gemini API that could be added to Banana Appeal:

1. **Aspect ratio and resolution control** - Let users specify output dimensions
2. **Model selection** - Choose between speed (Flash) and quality (Pro)
3. **Google Search grounding** - Generate images with real-time data
4. **Negative prompts** - Specify what to exclude from generation
5. **Image upscaling** - Enhance resolution of existing images
6. **Seed parameter** - Reproducible generation for iteration
