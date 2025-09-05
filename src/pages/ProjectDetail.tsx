import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Calendar, User, ChevronRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import ContactSection from '@/components/ContactSection';
import { CodePreview } from '@/components/CodePreview';
import { MathEquation } from '@/components/MathEquation';

// Enhanced project data structure for authoritative content
const projectsData = {
  "automated-valve-test": {
    id: "automated-valve-test",
    title: "Automated Valve Test Platform for High-Pressure Systems",
    subtitle: "Python-driven automation system for aerospace grade valve testing",
    category: "Test Automation",
    date: "2024",
    author: "Aerospace Engineering Team", 
    tags: ["High-Pressure Testing", "Python", "Automation"],
    hero: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png",
    sections: [
      {
        type: "text-left",
        title: "Context & Goal",
        content: `The Mars Entry Vehicle Heat Shield Analysis project addresses one of the most critical challenges in Mars exploration: protecting spacecraft during the intense heating phase of atmospheric entry. With entry velocities exceeding 6 km/s and peak heat fluxes reaching 1000 W/cm², the thermal protection system (TPS) must be precisely engineered to ensure mission success.
        
        Our objective was to develop a comprehensive computational model to predict heat shield performance during various entry scenarios, optimizing material selection and geometry for maximum protection while minimizing mass penalties. This analysis directly supports NASA's Mars Sample Return mission requirements and future crewed missions to Mars.`,
        visual: {
          type: "terminal",
          content: `// Mars Entry Parameters
Entry Velocity: 6.2 km/s
Entry Angle: -14.2°
Atmospheric Density: 0.020 kg/m³
Peak Heat Flux: 1,200 W/cm²
Entry Duration: 420 seconds

// Heat Shield Specifications
Diameter: 4.5 meters
Thickness: 5.8 cm (PICA-X)
Mass: 385 kg
Thermal Conductivity: 0.15 W/m·K`
        }
      },
      {
        type: "text-right",
        title: "Theoretical Background",
        content: `The heat transfer during Mars entry follows complex physics involving convective heating, radiation, and material ablation. The Fay-Riddell correlation provides the foundation for stagnation point heating calculations, while the Chapman-Rubesin parameter accounts for variable properties in the boundary layer.`,
        equations: [
          {
            equation: "q̇ = ρ∞U∞³√(ρw/ρ∞) × Pr^(-2/3)",
            variables: [
              { symbol: "q̇", description: "convective heat flux (W/cm²)" },
              { symbol: "ρ∞", description: "freestream density (kg/m³)" },
              { symbol: "U∞", description: "freestream velocity (m/s)" },
              { symbol: "Pr", description: "Prandtl number" }
            ]
          },
          {
            equation: "\\frac{∂T}{∂t} = α∇²T + \\frac{Q̇}{ρcp}",
            variables: [
              { symbol: "α", description: "thermal diffusivity (m²/s)" },
              { symbol: "Q̇", description: "heat generation rate (W/m³)" },
              { symbol: "cp", description: "specific heat capacity (J/kg·K)" }
            ]
          }
        ],
        codePreview: {
          title: "Fay-Riddell Stagnation Point Heating",
          preview: `import numpy as np

def fay_riddell_heating(rho_inf, u_inf, r_n, pr=0.71):
    """Calculate stagnation point heating rate"""
    C = 0.94  # Fay-Riddell constant
    q_dot = C * np.sqrt(rho_inf/r_n) * u_inf**3 * pr**(-2/3)
    return q_dot

# Mars entry conditions
rho_mars = 0.020  # kg/m³
velocity = 6200   # m/s
nose_radius = 2.25  # m

heating_rate = fay_riddell_heating(rho_mars, velocity, nose_radius)
print(f"Peak heating: {heating_rate:.2f} W/cm²")`,
          fullCode: `import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def fay_riddell_heating(rho_inf, u_inf, r_n, pr=0.71):
    """
    Calculate stagnation point heating rate using Fay-Riddell correlation
    
    Parameters:
    rho_inf: freestream density (kg/m³)
    u_inf: freestream velocity (m/s)
    r_n: nose radius (m)
    pr: Prandtl number
    
    Returns:
    q_dot: heating rate (W/cm²)
    """
    C = 0.94  # Fay-Riddell constant
    q_dot = C * np.sqrt(rho_inf/r_n) * u_inf**3 * pr**(-2/3)
    return q_dot

def heat_conduction_1d(T, t, alpha, thickness, q_surface):
    """
    1D heat conduction equation with surface heating
    """
    dTdt = np.zeros_like(T)
    dx = thickness / (len(T) - 1)
    
    # Interior points
    for i in range(1, len(T)-1):
        dTdt[i] = alpha * (T[i+1] - 2*T[i] + T[i-1]) / dx**2
    
    # Boundary conditions
    dTdt[0] = q_surface / (rho * cp * dx)  # Surface heating
    dTdt[-1] = 0  # Insulated back face
    
    return dTdt

# Mars entry trajectory data
time = np.linspace(0, 420, 1000)  # Entry duration: 420 seconds
altitude = 125000 - 3500 * time  # Linear approximation
velocity = 6200 * np.exp(-time/200)  # Velocity decay
rho_atm = 0.020 * np.exp((125000 - altitude)/10000)  # Exponential atmosphere

# Material properties (PICA-X)
rho = 280  # kg/m³
cp = 1500  # J/kg·K
k = 0.15   # W/m·K
alpha = k / (rho * cp)
thickness = 0.058  # m

# Calculate heating profile
nose_radius = 2.25  # m
heating_profile = []

for i, t in enumerate(time):
    q_dot = fay_riddell_heating(rho_atm[i], velocity[i], nose_radius)
    heating_profile.append(q_dot)

heating_profile = np.array(heating_profile)

# Solve heat conduction
n_nodes = 100
x = np.linspace(0, thickness, n_nodes)
T_initial = np.ones(n_nodes) * 300  # Initial temperature: 300 K

# Temperature evolution
T_solution = []
for i, q_surf in enumerate(heating_profile[::10]):  # Sample every 10 points
    if i == 0:
        T = T_initial
    else:
        t_span = [0, 4.2]  # 4.2 second intervals
        sol = odeint(heat_conduction_1d, T, t_span, args=(alpha, thickness, q_surf))
        T = sol[-1]
    T_solution.append(T.copy())

# Results analysis
max_surface_temp = max([T[0] for T in T_solution])
max_heating = max(heating_profile)

print(f"Maximum heating rate: {max_heating:.2f} W/cm²")
print(f"Maximum surface temperature: {max_surface_temp:.0f} K")
print(f"Peak temperature location: Surface")

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Heating profile
ax1.plot(time, heating_profile, 'r-', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Heat Flux (W/cm²)')
ax1.set_title('Convective Heating During Mars Entry')
ax1.grid(True, alpha=0.3)

# Velocity and altitude
ax2_twin = ax2.twinx()
ax2.plot(time, velocity/1000, 'b-', linewidth=2, label='Velocity')
ax2_twin.plot(time, altitude/1000, 'g-', linewidth=2, label='Altitude')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (km/s)', color='b')
ax2_twin.set_ylabel('Altitude (km)', color='g')
ax2.set_title('Entry Trajectory')
ax2.grid(True, alpha=0.3)

# Temperature distribution
time_samples = np.linspace(0, len(T_solution)-1, 5, dtype=int)
for i, t_idx in enumerate(time_samples):
    ax3.plot(x*1000, T_solution[t_idx], label=f't = {t_idx*42:.0f} s')
ax3.set_xlabel('Distance from surface (mm)')
ax3.set_ylabel('Temperature (K)')
ax3.set_title('Temperature Distribution Through Heat Shield')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Safety factor analysis
T_limit = 3000  # Material temperature limit (K)
safety_factor = T_limit / max_surface_temp
print(f"\\nSafety Analysis:")
print(f"Material temperature limit: {T_limit} K")
print(f"Maximum predicted temperature: {max_surface_temp:.0f} K")
print(f"Safety factor: {safety_factor:.2f}")

if safety_factor > 1.5:
    print("✓ Heat shield design meets safety requirements")
else:
    print("⚠ Heat shield design requires optimization")`
        }
      }
    ]
  },
  "rga-sensor-integration": {
    id: "rga-sensor-integration",
    title: "RGA Sensor Integration with Unitree Go2 Robot",
    subtitle: "Designed and simulated a robust mounting system to integrate a Residual Gas Analyzer sensor onto a quadruped robot",
    category: "Mechanical Design",
    date: "2024",
    author: "Engineering Team",
    tags: ["CAD Design", "Vibration Isolation", "Robotics Integration"],
    hero: "/lovable-uploads/7e9814d1-b051-4b58-99a9-b57a50fe4738.png",
    sections: [
      {
        type: "text-left",
        title: "Context & Goal",
        content: "PLACEHOLDER: This content needs to be updated with the actual project details from the Google Drive document.",
        visual: {
          type: "terminal",
          content: `// Placeholder content
// To be updated with real project data`
        }
      }
    ]
  },
  "uav-propulsion-optimization": {
    id: "uav-propulsion-optimization",
    title: "UAV Propulsion Optimization via High-Fidelity Simulation",
    subtitle: "Conducted advanced CFD, combustion, and acoustic simulations of UAV propulsion systems for performance optimization",
    category: "CFD Analysis",
    date: "2024",
    author: "Engineering Team",
    tags: ["ANSYS Fluent", "CFD Modeling", "LMS Virtual.Lab"],
    hero: "/lovable-uploads/8cf36141-768e-42d1-9dd6-1da18d8ddee5.png",
    sections: [
      {
        type: "text-left",
        title: "Context & Goal",
        content: "PLACEHOLDER: This content needs to be updated with the actual project details from the Google Drive document.",
        visual: {
          type: "terminal",
          content: `// Placeholder content
// To be updated with real project data`
        }
      }
    ]
  },
  "vibration-fatigue-detection": {
    id: "vibration-fatigue-detection",
    title: "Vibration-Based Fatigue Risk Detection for NASA's MSolo Mass Spectrometer",
    subtitle: "Developed real-time anomaly detection algorithms using FFT analysis and machine learning for fatigue risk assessment",
    category: "Signal Processing",
    date: "2024",
    author: "Engineering Team",
    tags: ["FFT Analysis", "Machine Learning", "Real-Time Detection"],
    hero: "/lovable-uploads/d1e74099-500d-4c46-a984-3fbe6f55a551.png",
    sections: [
      {
        type: "text-left",
        title: "Context & Goal",
        content: "PLACEHOLDER: This content needs to be updated with the actual project details from the Google Drive document.",
        visual: {
          type: "terminal",
          content: `// Placeholder content
// To be updated with real project data`
        }
      }
    ]
  },
  "uav-tail-fuselage": {
    id: "uav-tail-fuselage",
    title: "UAV Tail & Fuselage Variations for Stability Analysis",
    subtitle: "Performed comprehensive stability analysis of various UAV design configurations to optimize flight performance",
    category: "Aerodynamics",
    date: "2024",
    author: "Engineering Team",
    tags: ["Stability Analysis", "Flight Dynamics", "Design Optimization"],
    hero: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png",
    sections: [
      {
        type: "text-left",
        title: "Context & Goal",
        content: "PLACEHOLDER: This content needs to be updated with the actual project details from the Google Drive document.",
        visual: {
          type: "terminal",
          content: `// Placeholder content
// To be updated with real project data`
        }
      }
    ]
  }
};

const ProjectDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const project = projectsData[id as keyof typeof projectsData];

  if (!project) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="container mx-auto px-4 pt-24 text-center">
          <h1 className="text-4xl font-bold mb-4">Project Not Found</h1>
          <Button onClick={() => navigate('/projects')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Projects
          </Button>
        </div>
      </div>
    );
  }

  const renderSection = (section: any, index: number) => {
    const isEven = index % 2 === 0;
    const textFirst = section.type === "text-left" ? isEven : !isEven;

    return (
      <section 
        key={index} 
        className={`py-16 animate-fade-in`}
        style={{ animationDelay: `${index * 0.2}s` }}
      >
        <div className="container mx-auto px-4">
          <div className={`grid lg:grid-cols-2 gap-12 items-start ${textFirst ? '' : 'lg:grid-flow-col-dense'}`}>
            {/* Text Content */}
            <div className={`space-y-6 ${textFirst ? '' : 'lg:col-start-2'}`}>
              <div className="engineering-callout">
                <h2 className="text-3xl font-bold mb-6 text-primary">{section.title}</h2>
                <div className="prose prose-lg max-w-none text-foreground/90 leading-relaxed">
                  {section.content.split('\n\n').map((paragraph: string, pIndex: number) => (
                    <p key={pIndex} className="mb-4">
                      {paragraph}
                    </p>
                  ))}
                </div>
              </div>

              {/* Mathematical Equations */}
              {section.equations && section.equations.map((eq: any, eqIndex: number) => (
                <MathEquation
                  key={eqIndex}
                  equation={eq.equation}
                  variables={eq.variables}
                />
              ))}

              {/* Engineering Standards Callout */}
              {section.standards && (
                <div className="engineering-callout">
                  <h4 className="font-semibold mb-2">Engineering Standards</h4>
                  <ul className="space-y-1 text-sm">
                    {section.standards.map((standard: string, sIndex: number) => (
                      <li key={sIndex} className="flex items-center">
                        <ChevronRight className="w-4 h-4 mr-2 text-accent" />
                        <span className="metrics-badge">{standard}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Pull Quote */}
              {section.pullQuote && (
                <blockquote className="pull-quote">
                  "{section.pullQuote}"
                </blockquote>
              )}

              {/* Metrics */}
              {section.metrics && (
                <div className="flex flex-wrap gap-2">
                  {section.metrics.map((metric: any, mIndex: number) => (
                    <span key={mIndex} className="metrics-badge">
                      {metric.label}: {metric.value}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Visual/Code Content */}
            <div className={`${textFirst ? 'lg:col-start-2' : ''}`}>
              {section.visual && section.visual.type === "terminal" && (
                <div className="bg-[#0D1117] rounded-lg border border-gray-800 overflow-hidden shadow-lg">
                  <div className="flex items-center px-4 py-2 bg-[#21262d] border-b border-gray-800">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <span className="ml-4 text-sm text-gray-400">Terminal</span>
                  </div>
                  <div className="p-4">
                    <pre className="text-sm text-gray-100 font-mono whitespace-pre-wrap">
                      {section.visual.content}
                    </pre>
                  </div>
                </div>
              )}

              {section.codePreview && (
                <CodePreview
                  title={section.codePreview.title}
                  preview={section.codePreview.preview}
                  fullCode={section.codePreview.fullCode}
                />
              )}

              {section.image && (
                <div className="rounded-lg overflow-hidden shadow-lg">
                  <img 
                    src={section.image} 
                    alt={section.title}
                    className="w-full h-auto"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    );
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative pt-24 pb-16 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-background to-accent/10" />
        <div className="container mx-auto px-4 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <div className="flex items-center justify-center gap-4 mb-6">
              <Badge variant="secondary" className="text-sm">{project.category}</Badge>
              <div className="flex items-center text-sm text-muted-foreground">
                <Calendar className="w-4 h-4 mr-2" />
                {project.date}
              </div>
              <div className="flex items-center text-sm text-muted-foreground">
                <User className="w-4 h-4 mr-2" />
                {project.author}
              </div>
            </div>
            
            <h1 className="text-5xl font-bold mb-4 text-gradient">{project.title}</h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              {project.subtitle}
            </p>
            
            <div className="flex flex-wrap justify-center gap-2 mb-8">
              {project.tags.map((tag: string, index: number) => (
                <Badge key={index} variant="outline" className="text-sm">
                  {tag}
                </Badge>
              ))}
            </div>

            <Button 
              variant="outline" 
              onClick={() => navigate('/projects')}
              className="inline-flex items-center"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Projects
            </Button>
          </div>
        </div>
      </section>

      {/* Project Sections */}
      <div className="relative">
        {project.sections.map((section: any, index: number) => renderSection(section, index))}
      </div>

      {/* Back to Projects */}
      <section className="py-16 border-t border-border/50">
        <div className="container mx-auto px-4 text-center">
          <Button 
            onClick={() => navigate('/projects')}
            className="btn-primary"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Projects
          </Button>
        </div>
      </section>

      {/* Contact Section */}
      <ContactSection />
      
      {/* Footer */}
      <Footer />
    </div>
  );
};

export default ProjectDetail;
