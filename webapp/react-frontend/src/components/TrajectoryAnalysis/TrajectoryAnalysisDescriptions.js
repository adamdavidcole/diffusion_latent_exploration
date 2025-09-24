export const TrajectoryAnalysisDescriptions = {
    // TEMPORAL ANALYSIS
    trajectory_length: {
        "short_description": "Total distance traveled in latent space during diffusion.",
        "formula": "Length = Σₜ ||xₜ₊₁ - xₜ||",
        "formula_code": "step_differences = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]\nstep_norms = torch.linalg.norm(step_differences, dim=2)\ntrajectory_lengths = torch.sum(step_norms, dim=1)",
        "description": "Longer trajectories indicate more complex latent space exploration during generation. This metric reveals how much the latent representation changes throughout the diffusion process, with longer paths suggesting more extensive refinement and detail addition.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_temporal_trajectories.py#L14-L19"
    },

    velocity_analysis: {
        "short_description": "Speed of movement through latent space per diffusion step.",
        "formula": "vₜ = ||xₜ₊₁ - xₜ||, mean_velocity = Σvₜ/T",
        "formula_code": "step_differences = flat_trajectories[:, 1:] - flat_trajectories[:, :-1]\nvelocities = torch.linalg.norm(step_differences, dim=2)\nmean_velocity = torch.mean(velocities, dim=1)\nvelocity_variance = torch.var(velocities, dim=1)",
        "description": "Mean velocity indicates average latent change rate; velocity variance reveals whether speed is steady or episodic. High variance suggests phase transitions - periods of rapid change followed by stability. Consistent velocities indicate smooth generation while variable speeds mark critical generation moments.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_temporal_trajectories.py#L22-L30"
    },

    acceleration_analysis: {
        "short_description": "Rate of change of velocity between steps.",
        "formula": "aₜ = ||vₜ₊₁ - vₜ||, mean_acceleration = Σaₜ/T",
        "formula_code": "velocities = torch.linalg.norm(step_differences, dim=2)\naccel_differences = velocities[:, 1:] - velocities[:, :-1]\naccel_norms = torch.linalg.norm(accel_differences, dim=2)\nmean_acceleration = torch.mean(accel_norms, dim=1)",
        "description": "High acceleration indicates 'jolts' in the generative process - moments where the model rapidly changes its rate of latent evolution. Acceleration spikes often correspond to phase transitions where the model shifts between different types of processing (e.g., from removing noise to adding fine details).",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_temporal_trajectories.py#L32-L40"
    },

    tortuosity: {
        "short_description": "Measures path indirectness - how winding the trajectory is.",
        "formula": "circuitousness = trajectory_length / endpoint_distance",
        "formula_code": "trajectory_lengths = torch.sum(step_norms, dim=1)\nendpoint_distances = torch.linalg.norm(flat_trajectories[:, -1] - flat_trajectories[:, 0], dim=1)\ntortuosity = trajectory_lengths / (endpoint_distances + 1e-8)",
        "description": "A value of 1.0 indicates a perfectly straight trajectory from start to finish. Higher values indicate meandering paths. This reveals generation efficiency - direct paths suggest focused generation toward a clear goal, while winding paths indicate exploration, uncertainty, or complex multi-stage refinement processes.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_individual_trajectory_geometry.py#L64-L65"
    },

    endpoint_distance: {
        "short_description": "Straight-line distance between initial and final latent states.",
        "formula": "||x_final - x_initial||",
        "formula_code": "endpoint_distances = torch.linalg.norm(flat_trajectories[:, -1] - flat_trajectories[:, 0], dim=1)",
        "description": "This measures total displacement in latent space, independent of path taken. Always ≤ trajectory length, with their ratio defining circuitousness. Large endpoint distances indicate the final output differs significantly from initial noise, while small distances suggest minimal semantic transformation during generation.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_temporal_trajectories.py#L42-L45"
    },

    semantic_convergence: {
        "short_description": "Step at which trajectory reaches half the initial distance to final state.",
        "formula": "half_life = argmin(t) where ||xₜ - x_final|| ≤ 0.5 * ||x₀ - x_final||",
        "formula_code": "final_latents = flat_trajectories[:, -1, :].unsqueeze(1)\ndistances_to_end = torch.linalg.norm(flat_trajectories - final_latents, dim=2)\nhalf_distance = distances_to_end[:, 0] / 2.0\nhalf_life_step = torch.argmax(half_life_mask.int(), dim=1)",
        "description": "This half-life indicates when trajectories are 'halfway' to their final semantic state. Smaller values mean most change happens early (remaining steps are refinements), while larger values indicate significant changes continue into later steps. Reveals the temporal structure of the generation process.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_temporal_trajectories.py#L69-L85"
    },

    // GEOMETRIC ANALYSIS
    speed_stats: {
        "short_description": "Individual trajectory movement magnitude per step.",
        "formula": "speed = ||xₜ₊₁ - xₜ||",
        "formula_code": "step_differences = trajectory[1:] - trajectory[:-1]\nstep_sizes = np.linalg.norm(step_differences, axis=1)\nmean_speed = np.mean(step_sizes)",
        "description": "This is equivalent to the temporal analysis velocity but computed per individual trajectory rather than aggregated. Speed quantifies how rapidly each trajectory evolves through latent space, providing insight into per-prompt generation dynamics and individual variation patterns."
    },

    log_volume_stats: {
        "short_description": "Logarithm of trajectory bounding box volume in latent space.",
        "formula": "log_volume = Σᵢ log(maxᵢ - minᵢ)",
        "formula_code": "mins = np.min(trajectory, axis=0)\nmaxs = np.max(trajectory, axis=0)\nranges = np.clip(maxs - mins, 1e-12, None)\nlog_bbox_vol = np.sum(np.log(ranges))\neff_side = float(np.exp(log_bbox_vol / ranges.size))",
        "description": "This measures the geometric extent of the trajectory's exploration through high-dimensional latent space. Larger values indicate trajectories that span greater ranges across latent dimensions, suggesting more diverse or exploratory generation processes."
    },

    effective_side_stats: {
        "short_description": "Geometric mean of trajectory bounding box dimensions.",
        "formula": "effective_side = exp(log_volume / D)",
        "formula_code": "log_bbox_vol = np.sum(np.log(ranges))\neff_side = np.exp(log_bbox_vol / ranges.size)",
        "description": "This provides a characteristic length scale for the trajectory's spatial extent, representing an average 'side length' of the high-dimensional bounding region explored by the trajectory. It normalizes the volume measure to a single dimension for easier interpretation."
    },

    endpoint_alignment_stats: {
        "short_description": "Alignment between step directions and overall trajectory direction.",
        "formula": "alignment = mean((vₜ · e) / (||vₜ|| · ||e||))",
        "formula_code": "e = trajectory[-1] - trajectory[0]\ne_norm = np.linalg.norm(e) + 1e-8\nv = step_differences\nv_norms = np.linalg.norm(v, axis=1) + 1e-8\nendpoint_alignment = np.mean((v @ e) / (v_norms * e_norm))",
        "description": "Values near 1.0 indicate steps consistently move toward the endpoint; lower values suggest deviation from the direct path. This metric complements tortuosity by focusing on directional consistency rather than path length efficiency."
    },

    turning_angle_stats: {
        "short_description": "Total angular change along trajectory path.",
        "formula": "turning_angle = Σₜ arccos(uₜ · uₜ₊₁)",
        "formula_code": "u = v[:-1] / v_norms[:-1, None]\nw = v[1:] / v_norms[1:, None]\ncosang = np.clip(np.sum(u*w, axis=1), -1.0, 1.0)\nturning_angle = np.sum(np.arccos(cosang))",
        "description": "Large turning angles indicate sharp directional changes; small angles suggest smooth, consistent movement through latent space. This metric helps identify whether trajectories follow smooth paths or make abrupt directional changes during generation."
    },

    circuitousness_stats: {
        "short_description": "Path efficiency measure from individual trajectory analysis.",
        "formula": "circuitousness = path_length / endpoint_distance",
        "formula_code": "path_length = np.sum(step_sizes)\nendpoint_distance = np.linalg.norm(trajectory[-1] - trajectory[0])\ncircuitousness = path_length / (endpoint_distance + 1e-8)",
        "description": "This is equivalent to the temporal analysis tortuosity but computed at the individual trajectory level. Values close to 1.0 indicate direct paths; larger values indicate meandering exploration. Individual analysis reveals the distribution of path efficiency across different generation runs."
    },

    efficiency_metrics: {
        "short_description": "Summary statistics of trajectory path efficiency.",
        "formula": "efficiency = 1 / circuitousness",
        "formula_code": "mean_efficiency = np.mean([1.0/c for c in circuitousness])\nballistic_count = np.sum([c < 1.5 for c in circuitousness])\nmeandering_count = np.sum([c > 3.0 for c in circuitousness])",
        "description": "Higher efficiency indicates more direct navigation through latent space, while lower efficiency suggests exploratory behavior. The counts of ballistic vs meandering trajectories reveal the distribution of generation strategies within each prompt group."
    },

    step_variability_stats: {
        "short_description": "Consistency of step sizes throughout trajectories.",
        "formula": "variability = std(step_sizes)",
        "formula_code": "step_sizes = np.linalg.norm(step_differences, axis=1)\nspeed_variability = np.std(step_sizes)",
        "description": "High variability indicates irregular, episodic movement with varying step sizes; low variability suggests steady, regular progression through latent space. This metric reveals whether the generation process maintains consistent pacing or exhibits burst-like behavior."
    },

    // GEOMETRY DERIVATIVES
    curvature_peak_mean: {
        "short_description": "Maximum curvature along trajectory paths.",
        "formula": "curvatureₜ = ||Δvₜ|| / (||vₜ|| + ε)",
        "formula_code": "v = X[:, 1:, :] - X[:, :-1, :]\ndv = v[:, 2:, :] - v[:, 1:-1, :]\ncurv = dv.norm(dim=2) / (v[:, 2:, :].norm(dim=2) + eps)\nkmax, kidx = curv.max(dim=1)",
        "description": "Curvature quantifies how much the trajectory direction changes relative to distance traveled - essentially detecting sharp turns. High peak curvature often aligns with phase transitions where the model shifts focus (e.g., from broad structure to specific details). More complex prompts typically show higher peak curvature and later timing.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_geometry_derivatives.py#L20-L31"
    },

    jerk_peak_mean: {
        "short_description": "Maximum jerk (third derivative) along trajectory paths.",
        "formula": "jerkₜ = ||Δaₜ|| (change in acceleration)",
        "formula_code": "a = v[:, 1:, :] - v[:, :-1, :]\nj = a[:, 1:, :] - a[:, :-1, :]\njerk = j.norm(dim=2)\njmax, jidx = jerk.max(dim=1)",
        "description": "Jerk measures sudden changes in acceleration - moments when the model abruptly shifts from slowing down to speeding up (or vice versa). High jerk indicates dramatic transitions in generation rate, potentially marking critical decision points where the model's attention shifts between different aspects of the output.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_geometry_derivatives.py#L25-L40"
    },

    // SPATIAL ANALYSIS
    trajectory_pattern: {
        "short_description": "Evolution of spatial variance over diffusion steps.",
        "formula": "spatial_pattern = mean(var(trajectories, dim=spatial))",
        "formula_code": "spatial_vars_per_step = torch.var(trajectories, dim=(-2, -1))\nspatial_vars_mean_per_step = torch.mean(spatial_vars_per_step, dim=(2, 3))\ngroup_spatial_trajectory = torch.mean(spatial_vars_mean_per_step, dim=0)",
        "description": "This tracks how spatial detail evolves throughout diffusion, often showing a U-shaped curve where spatial variance first decreases (noise removal) then increases (detail emergence). The trajectory pattern reveals the temporal dynamics of spatial structure development during generation.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_spatial_patterns.py#L32-L39"
    },

    evolution_ratio: {
        "short_description": "Late-stage to early-stage spatial variance ratio.",
        "formula": "ratio = late_spatial_mean / early_spatial_mean",
        "formula_code": "early_steps = spatial_vars_mean_per_step[:, :n_steps//3]\nlate_steps = spatial_vars_mean_per_step[:, -n_steps//3:]\nearly_spatial_mean = torch.mean(early_steps)\nlate_spatial_mean = torch.mean(late_steps)\nspatial_evolution_ratio = late_spatial_mean / (early_spatial_mean + 1e-8)",
        "description": "Ratios > 1 indicate detail recovery/emergence in late stages following the classic coarse-to-fine generation pattern. Higher ratios (often seen with complex prompts) suggest the latent first loses spatial variance (noise removal) then gains variance as structured details appear. This captures the U-shaped spatial evolution curve.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_spatial_patterns.py#L50-L56"
    },

    early_vs_late_significance: {
        "short_description": "Magnitude of difference between early and late spatial patterns.",
        "formula": "|late_spatial_mean - early_spatial_mean|",
        "formula_code": "early_spatial_mean = torch.mean(early_steps)\nlate_spatial_mean = torch.mean(late_steps)\nearly_vs_late_significance = torch.abs(early_spatial_mean - late_spatial_mean)",
        "description": "This quantifies how dramatically spatial patterns change between early and late diffusion phases, indicating the strength of the coarse-to-fine transition in the generation process. Larger values suggest more pronounced phase transitions."
    },

    trajectory_smoothness: {
        "short_description": "Mean absolute change in spatial variance between consecutive steps.",
        "formula": "smoothness = mean(|Δspatial_variance|)",
        "formula_code": "spatial_trajectory_deltas = torch.diff(spatial_vars_mean_per_step, dim=1)\ntrajectory_smoothness = torch.mean(torch.abs(spatial_trajectory_deltas))",
        "description": "Lower values indicate smooth spatial evolution; higher values suggest abrupt transitions or phase changes in spatial detail development. This metric reveals whether spatial evolution is gradual or contains sudden jumps."
    },

    phase_transition_strength: {
        "short_description": "Variability in spatial variance trajectory indicating phase transition intensity.",
        "formula": "strength = std(spatial_trajectory)",
        "formula_code": "group_spatial_trajectory = torch.mean(spatial_vars_mean_per_step, dim=0)\nphase_transition_strength = torch.std(group_spatial_trajectory)",
        "description": "High values indicate strong phase transitions with abrupt changes where the model shifts from removing noise to adding structured details. Low values suggest gradual, smooth spatial evolution. This metric identifies whether spatial detail emerges suddenly (sharp phase transition) or gradually throughout diffusion.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_spatial_patterns.py#L39-L40"
    },

    step_deltas_mean: {
        "short_description": "Mean change in spatial variance at each step across videos.",
        "formula": "step_deltas = mean(Δspatial_variance, dim=videos)",
        "formula_code": "spatial_trajectory_deltas = torch.diff(spatial_vars_mean_per_step, dim=1)\nstep_deltas_mean = torch.mean(spatial_trajectory_deltas, dim=0)",
        "description": "This reveals when spatial detail typically increases or decreases during diffusion, helping identify critical phases in the generation process. The step-by-step pattern shows the temporal dynamics of spatial evolution."
    },

    step_deltas_std: {
        "short_description": "Variability in spatial changes across videos at each step.",
        "formula": "step_deltas_std = std(Δspatial_variance, dim=videos)",
        "formula_code": "spatial_trajectory_deltas = torch.diff(spatial_vars_mean_per_step, dim=1)\nstep_deltas_std = torch.std(spatial_trajectory_deltas, dim=0)",
        "description": "High std at certain steps indicates those are transition points where different trajectories may follow different spatial evolution paths. This reveals the consistency of the generation process across different runs."
    },

    progression_consistency: {
        "short_description": "Cross-video spatial consistency at each step.",
        "formula": "consistency = mean(std(spatial_variance_across_videos))",
        "formula_code": "step_consistency = torch.std(spatial_vars_mean_per_step, dim=0)\nprogression_consistency = torch.mean(step_consistency)",
        "description": "Measures how consistently different videos (seeds) develop spatially at each step. Low values indicate all videos follow similar spatial evolution patterns; high values suggest divergent spatial development. Reveals whether the prompt produces deterministic or variable spatial generation paths."
    },

    progression_variability: {
        "short_description": "How much cross-video consistency varies across diffusion steps.",
        "formula": "variability = std(consistency_across_steps)",
        "formula_code": "step_consistency = torch.std(spatial_vars_mean_per_step, dim=0)\nprogression_variability = torch.std(step_consistency)",
        "description": "This captures whether the generation process maintains uniform consistency throughout or has phases of convergent vs divergent behavior. High variability suggests some steps produce very similar outputs across videos while others produce highly variable results."
    },

    inter_video_diversity_mean: {
        "short_description": "Mean spatial variability within individual videos over time.",
        "formula": "diversity_mean = mean(std(spatial_variance_per_video, dim=time))",
        "formula_code": "video_spatial_diversity = torch.std(spatial_vars_mean_per_step, dim=1)\ninter_video_diversity_mean = torch.mean(video_spatial_diversity)",
        "description": "This measures how much spatial detail fluctuates within each individual video throughout the diffusion process. Higher values indicate videos where spatial patterns vary dramatically over time, while lower values suggest more stable spatial evolution within each video generation.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_spatial_patterns.py#L44-L45"
    },

    inter_video_diversity_std: {
        "short_description": "Consistency of spatial variability patterns across different videos.",
        "formula": "diversity_std = std(std(spatial_variance_per_video, dim=time))",
        "formula_code": "video_spatial_diversity = torch.std(spatial_vars_mean_per_step, dim=1)\ninter_video_diversity_std = torch.std(video_spatial_diversity)",
        "description": "This measures how consistently different videos exhibit spatial variability. Low values indicate all videos have similar levels of internal spatial variation; high values suggest some videos are much more spatially dynamic than others. Reveals the consistency of the generation process across different seeds.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_spatial_patterns.py#L44-L45"
    },

    // STRUCTURAL VARIANCE METRICS
    temporal_variance: {
        "short_description": "Variance across time steps for each video.",
        "formula": "temporal_var = var(trajectory, dim=time)",
        "formula_code": "temporal_variance = torch.var(flat_trajectories, dim=1)\nmean_temporal_variance = torch.mean(temporal_variance, dim=1)",
        "description": "This measures how much each trajectory varies over time, indicating the magnitude of change during the diffusion process. Higher values suggest more dynamic evolution, while lower values indicate more stable trajectories."
    },

    spatial_variance: {
        "short_description": "Variance across videos for each step.",
        "formula": "spatial_var = var(trajectories, dim=videos)",
        "formula_code": "spatial_variance = torch.var(flat_trajectories, dim=0)\nmean_spatial_variance = torch.mean(spatial_variance, dim=1)",
        "description": "This shows how much trajectories diverge from each other at each step, revealing phases where generation becomes more or less deterministic. High variance indicates diverse generation paths, low variance suggests convergent behavior."
    },

    overall_variance: {
        "short_description": "Total variance across all trajectory data.",
        "formula": "overall_var = var(all_data)",
        "formula_code": "all_data = flat_trajectories.reshape(-1, flat_trajectories.shape[-1])\noverall_variance = torch.var(all_data, dim=0)\nmean_overall_variance = torch.mean(overall_variance)",
        "description": "This provides a global measure of variability in the latent trajectory dataset, capturing the overall diversity and richness of the latent space exploration across all prompts and videos."
    },

    variance_across_videos: {
        "short_description": "Variance in trajectory averages across different video runs.",
        "formula": "var_videos = var(mean(trajectories, dim=time))",
        "formula_code": "trajectory_means = torch.mean(flat_trajectories, dim=1)\nvariance_across_videos = torch.var(trajectory_means)",
        "description": "This measures how much the overall trajectory characteristics vary between different generation seeds/runs for the same prompt, indicating the consistency or diversity of the generation process."
    },

    variance_across_steps: {
        "short_description": "Variance in step averages across diffusion timeline.",
        "formula": "var_steps = var(mean(trajectories, dim=videos))",
        "formula_code": "step_means = torch.mean(flat_trajectories, dim=0)\nvariance_across_steps = torch.var(step_means)",
        "description": "This measures how much the average latent state varies across different diffusion steps, indicating the temporal dynamics of the generation process and the magnitude of change throughout diffusion."
    },

    // SCATTER PLOT RELATIONSHIPS
    velocity_vs_log_volume: {
        "short_description": "Relationship between trajectory speed and spatial extent.",
        "formula": "Scatter plot: (velocity, log_volume)",
        "formula_code": "# Combined from individual trajectory analysis\n# Each point: (mean_speed, log_bbox_vol)\nvelocity_per_traj = [np.mean(step_sizes) for trajectory in trajectories]\nlog_volume_per_traj = [np.sum(np.log(ranges)) for trajectory in trajectories]",
        "description": "Each point represents a single trajectory, colored by prompt group. This visualization reveals correlations between movement speed and spatial exploration, showing whether faster-moving trajectories tend to occupy larger or smaller volumes in latent space."
    },

    velocity_vs_circuitousness: {
        "short_description": "Relationship between trajectory speed and path efficiency.",
        "formula": "Scatter plot: (velocity, circuitousness - 1)",
        "formula_code": "# Combined from individual trajectory analysis\n# Each point: (mean_speed, circuitousness - 1.0)\nvelocity_per_traj = [np.mean(step_sizes) for trajectory in trajectories]\ncircuitousness_per_traj = [path_length/endpoint_dist - 1.0 for trajectory in trajectories]",
        "description": "Each point represents a single trajectory, with circuitousness adjusted by subtracting 1.0 (so 0 represents a straight line). This plot reveals whether high-velocity trajectories tend to follow direct or winding paths through latent space."
    },

    // CORRIDOR METRICS
    width_by_step: {
        "short_description": "Width of trajectory 'corridor' at each diffusion step.",
        "formula": "width_t = ||std_t||₂ where std_t is per-dimension standard deviation across trajectories",
        "formula_code": "mu = X.mean(dim=0)  # [T,D] centroid path\nstd = (X - mu.unsqueeze(0)).pow(2).mean(dim=0).sqrt()  # [T,D] per-dim std\nwidth = std.norm(dim=1)  # [T] L2 norm of std vector",
        "description": "Measures how tightly clustered trajectories are at each step. Lower values indicate trajectories follow similar paths (narrow corridor), while higher values show diverse paths (wide corridor). Evolution of width reveals whether generation becomes more or less deterministic over time.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_corridor_metrics.py#L35-L37"
    },

    branch_divergence: {
        "short_description": "Distance between each group's centroid and the baseline centroid at each step.",
        "formula": "divergence_t = ||μ_group,t - μ_baseline,t||₂",
        "formula_code": "base_centroid = flat[base].mean(dim=0)  # [T,D] baseline path\nmu = X.mean(dim=0)  # [T,D] group centroid path\nbranch = (mu - base_centroid).norm(dim=1)  # [T] distance per step",
        "description": "Tracks how far each experimental group deviates from the baseline condition throughout the diffusion process. Shows when and how much different conditions cause trajectories to diverge. Early divergence suggests prompt effects dominate early generation, while late divergence indicates refinement differences.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_corridor_metrics.py#L38-L39"
    },

    exit_distance: {
        "short_description": "Total cumulative divergence from baseline across all diffusion steps.",
        "formula": "exit_distance = Σₜ ||μ_group,t - μ_baseline,t||₂",
        "formula_code": "branch = (mu - base_centroid).norm(dim=1)  # [T] per-step divergence\nexit_distance = float(branch.sum().item())  # scalar total",
        "description": "Single scalar metric summarizing overall separation from baseline condition. Higher values indicate experimental conditions that consistently push trajectories away from baseline behavior. Useful for ranking conditions by their overall impact on latent space exploration patterns.",
        "source_url": "https://git.arts.ac.uk/acole/diffusion_latent_exploration/blob/main/src/analysis/latent_trajectory_analysis/analyze_corridor_metrics.py#L42"
    }
};