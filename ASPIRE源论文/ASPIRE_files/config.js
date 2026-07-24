/**
 * ASPIRE website configuration
 */
const SITE_CONFIG = {
  paperTitle: "ASPIRE: Agentic /Skills Discovery for Robotics",

  introText:
    "Traditional robot programming is notoriously challenging: it requires orchestrating multimodal perception, managing complex physical contact dynamics, and handling diverse environment configurations and execution failures. We introduce ASPIRE (Agentic Skill Programming through Iterative Robot Exploration), a continual learning system for robotics that autonomously writes and refines robot control programs in a code-as-policy paradigm while compounding experience into a reusable skill library. ASPIRE enables automated discovery of reusable skills that persist across multiple tasks, simulation and real-world settings, and different embodiments. Rather than relying on fixed, human-engineered pipelines, ASPIRE operates in an open-ended learning loop, consisting of three key components: (1) a closed-loop robot execution engine that exposes fine-grained multimodal traces (e.g., perception overlays, grasp candidates, motion trajectories, and collision feedback), enabling the agent to autonomously diagnose failures, synthesize repairs, and validate outcomes; (2) a continually expanding skill library that distills validated fixes into reusable, transferable robotic knowledge; and (3) an evolutionary search procedure that generates diverse task sequences and control programs, systematically debugging them to explore beyond single-trajectory refinement. As ASPIRE encounters more tasks, its growing skill library enables increasingly rapid adaptation. Consequently, ASPIRE surpasses prior methods by up to 77% on manipulation tasks under perturbation (LIBERO-Pro), 72% on Robosuite's bimanual handover task, and up to 32% on long-horizon household tasks (BEHAVIOR-1K). The accumulated skill library further enables strong zero-shot generalization: on representative unseen long-horizon tasks (LIBERO-Pro Long), ASPIRE achieves 31% success, substantially outperforming the 4% success rate of prior methods despite their heavy reliance on test-time reasoning and retries. Finally, skills discovered in simulation provide initial evidence of sim-to-real transfer, substantially reducing real-robot programming effort despite different embodiments and robot APIs.",

  teamName: "NVIDIA GEAR Team",
  teamUrl: "https://research.nvidia.com/labs/gear/",

  links: {
    arxiv: "#",
    paper: "assets/Aspire.pdf?v=20260630d",
    code: "",
  },

  gtagId: null,

  anonymous: false,

  authorHtml: `
    <span class="author-line">
      <span class="author-name">Runyu Lu<sup>1,2,*,†</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Yubo Wu<sup>1,3,*</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Ethan Kou<sup>1,4,*</sup></span>
    </span>
    <span class="author-line">
      <span class="author-name">Max Fu<sup>1,4</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Wenli Xiao<sup>1,5</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Ajay Mandlekar<sup>1</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Yinzhen Xu<sup>1</sup></span>
    </span>
    <span class="author-line">
      <span class="author-name">Guanya Shi<sup>5</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Ken Goldberg<sup>4</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Ang Chen<sup>2</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Mosharaf Chowdhury<sup>2</sup></span>
    </span>
    <span class="author-line">
      <span class="author-name">Yuke Zhu<sup>1,†</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Linxi “Jim” Fan<sup>1,†</sup></span> <span class="author-sep">·</span>
      <span class="author-name">Guanzhi Wang<sup>1,†</sup></span>
    </span>
    <span class="affiliation-line">
      <sup>1</sup>NVIDIA <span class="author-sep">·</span>
      <sup>2</sup>UMich <span class="author-sep">·</span>
      <sup>3</sup>UIUC <span class="author-sep">·</span>
      <sup>4</sup>UC Berkeley <span class="author-sep">·</span>
      <sup>5</sup>CMU
    </span>
    <span class="author-note-line"><sup>*</sup>Equal contribution <span class="author-sep">·</span> <sup>†</sup>Project leads</span>
  `,

  authors: [
    { name: "Runyu Lu", affiliations: [1, 2] },
    { name: "Yubo Wu", affiliations: [1, 3] },
    { name: "Ethan Kou", affiliations: [1, 4] },
    { name: "Max Fu", affiliations: [1, 4] },
    { name: "Wenli Xiao", affiliations: [1, 5] },
    { name: "Ajay Mandlekar", affiliations: [1] },
    { name: "Yinzhen Xu", affiliations: [1] },
    { name: "Guanya Shi", affiliations: [5] },
    { name: "Ken Goldberg", affiliations: [4] },
    { name: "Ang Chen", affiliations: [2] },
    { name: "Mosharaf Chowdhury", affiliations: [2] },
    { name: "Yuke Zhu", affiliations: [1] },
    { name: "Linxi “Jim” Fan", affiliations: [1] },
    { name: "Guanzhi Wang", affiliations: [1] },
  ],

  affiliations: {
    1: "NVIDIA",
    2: "UMich",
    3: "UIUC",
    4: "UC Berkeley",
    5: "CMU",
  },

  institutionLogos: [],
};
