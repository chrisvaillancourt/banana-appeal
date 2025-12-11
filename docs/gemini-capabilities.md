# Gemini Image Generation Capabilities

This document describes the image generation capabilities available through the Gemini API and how they're exposed (or could be exposed) through Banana Appeal.

## Available Models

| Model | Nickname | Best For | Max Images | Features |
|-------|----------|----------|------------|----------|
| `gemini-2.5-flash-image` | nano-banana | Fast generation, simple edits | 3 | Quick, cheap, good for iteration |
| `gemini-3-pro-image-preview` | nano-banana-pro | Complex tasks, high quality | 14 (6 high-fidelity) | 4K, text rendering, search grounding, thinking |

**Current default:** `gemini-2.5-flash-image`

**Recommendation:** Switch to `gemini-3-pro-image-preview` for:
- Text-heavy images (diagrams, infographics)
- 2K/4K resolution needs
- Real-time data (weather, current events)
- Complex multi-image compositions (>3 images)

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

Combine multiple reference images with a creative prompt:

| Model | Max Images | High-Fidelity Limit |
|-------|------------|---------------------|
| Flash | 3 | 3 |
| Pro | 14 | 6 |

Use cases:
- Style transfer from reference images
- Character consistency across generations
- Product compositions with multiple items
- Object + human reference images

**Tip:** To exceed the image limit, combine multiple images into a single collage first, then pass it as one image.

### Structured Responses

When saving images to disk, all tools return a structured response with metadata:

```json
{
  "path": "/tmp/image.jpg",
  "format": "jpeg",
  "warnings": []
}
```

**Extension Auto-Correction:** If you request a `.png` extension but Gemini returns JPEG, the file is saved with the correct `.jpg` extension and a warning is included:

```json
{
  "path": "/tmp/image.jpg",
  "format": "jpeg",
  "warnings": ["Gemini returned JPEG image; saved as .jpg (requested .png)"],
  "original_path": "/tmp/image.png"
}
```

**Verbose Mode:** Pass `verbose=True` to include additional metadata:

```json
{
  "path": "/tmp/image.jpg",
  "format": "jpeg",
  "warnings": [],
  "dimensions": {"width": 1376, "height": 768},
  "size_bytes": 689152,
  "generation_time_ms": 3420.5,
  "model": "gemini-2.5-flash-image"
}
```

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

**Default behavior:** The model matches input image dimensions when editing. For generation without input images, it defaults to 1:1.

### Resolution and Token Costs (Pro Model)

| Aspect Ratio | 1K Resolution | 2K Resolution | 4K Resolution |
|--------------|---------------|---------------|---------------|
| 1:1 | 1024x1024 | 2048x2048 | 4096x4096 |
| 16:9 | 1376x768 | 2752x1536 | 5504x3072 |
| 9:16 | 768x1376 | 1536x2752 | 3072x5504 |
| 3:2 | 1264x848 | 2528x1696 | 5056x3392 |

**Token costs:**
- 1K and 2K: ~1210 tokens (same price)
- 4K: ~2000 tokens (more expensive)

### Search Grounding Notes

When using Google Search grounding with Pro model:
- Must include both `TEXT` and `IMAGE` in `response_modalities`
- Image-only mode doesn't work with grounding
- Only text search results are used (not image search results)

## Creative Use Cases

Beyond coding workflows, Gemini excels at these creative tasks:

### Sprite Sheets for Games

```
Sprite sheet of a jumping character, 3x3 grid, white background,
sequence, frame by frame animation, square aspect ratio.
```

### Photo Colorization

```
Restore and colorize this historical black and white photograph from 1932.
```

### Style Transfer

```
Create a version of this image as if they were living in the 1980s,
capturing the fashion, hairstyles, and atmosphere of that era.
```

### Sticker Creation

```
Create a single sticker in Pop Art style. Bold, thick black outlines.
Vibrant primary colors in flat, unshaded blocks. Include visible
Ben-Day dots for texture.
```

### Mini-Figurine Generation

```
Create a 1/7 scale collectible figurine of this character in realistic style,
placed on a desk next to its packaging box designed like high-quality collectibles.
```

### Map to View Transformation

```
Show me what we would see from this location marked on the map.
```

## Chat Mode (Recommended for Editing)

For iterative image editing, chat/multi-turn mode is recommended over single calls. The SDK maintains context automatically through "thought signatures."

```python
chat = client.chats.create(model="gemini-2.5-flash-image")

# First generation
response = chat.send_message("Create a fox figurine in a bedroom")

# Iterative refinements - model remembers the fox
response = chat.send_message("Add a blue planet on its helmet")
response = chat.send_message("Move it to a beach setting")
response = chat.send_message("Now it should be cooking a barbecue")
```

**Benefits of chat mode:**
- Character consistency across edits
- Context preservation (model remembers previous images)
- More natural iterative workflow

## Future Enhancements

Features from the Gemini API that could be added to Banana Appeal:

1. **Google Search grounding** - Generate images with real-time data (requires Pro model)
2. **Image upscaling** - Enhance resolution of existing images via UpscaleImageConfig
3. **Chat/multi-turn mode** - Maintain context across multiple edits
4. **Thinking config** - Access Pro model's reasoning process

### Already Implemented

- **Aspect ratio control** - 10 supported ratios (1:1, 16:9, 9:16, etc.)
- **Resolution control** - 1K, 2K, or 4K output
- **Seed parameter** - Reproducible generation for iteration
- **Model-aware blend limits** - Flash models limited to 3 images, Pro to 14

### Not Available with Gemini Models

- **Negative prompts** - Only supported with Imagen models (via `generate_image` API), not Gemini's native image generation (`generate_content` API)

## References

Official documentation and resources used to compile this guide:

- [Gemini 3 Developer Guide](https://ai.google.dev/gemini-api/docs/gemini-3) - Model capabilities and thinking features
- [Image Generation Documentation](https://ai.google.dev/gemini-api/docs/image-generation) - API reference and prompting strategies
- [Image Understanding Documentation](https://ai.google.dev/gemini-api/docs/image-understanding) - Input formats and analysis capabilities
- [Gemini Cookbook](https://github.com/google-gemini/cookbook) - Official examples and notebooks
- [Native Image Generation Notebook](https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_Started_Nano_Banana.ipynb) - Comprehensive examples for nano-banana models
