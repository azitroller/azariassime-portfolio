import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Calendar, User, ChevronRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import ContactSection from '@/components/ContactSection';
import { MathEquation } from '@/components/MathEquation';

// Enhanced project data structure for authoritative content
const projectsData = {
  "automated-valve-test": {
    id: "automated-valve-test",
    title: "Automated Valve Test Platform",
    subtitle: "High-Pressure, High-Temperature Testing System for Aerospace Applications",
    category: "Systems Engineering",
    date: "Spring 2024",
    author: "Azarias Thomas",
    tags: ["Automation", "Testing", "Python", "Data Acquisition", "Aerospace", "INFICON"],
    hero: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png",
    sections: [
      {
        type: "overview",
        title: "Context & Goal",
        content: "During my internship at INFICON, I encountered a critical engineering challenge that would fundamentally reshape how the company approaches valve testing for aerospace applications. The existing manual testing protocol for high-pressure, high-temperature valve systems presented numerous limitations that compromised both data quality and operational efficiency in ways that extended far beyond simple inconvenience.\n\nOperators were required to maintain continuous presence during test cycles lasting 12-48 hours, manually recording pressure and temperature measurements at 5-10 minute intervals while managing complex systems operating at pressures up to 15,000 psi and temperatures ranging from -40°C to 180°C. This approach introduced significant data gaps during which critical transient events could occur undetected—a particularly concerning issue given that valve failure initiation in aerospace systems can happen within seconds, making these measurement gaps unacceptable blind spots in the testing protocol.\n\nThe human factor complications extended well beyond simple data collection issues and represented a fundamental limitation in the reliability of test results. Statistical analysis of historical manual test data revealed measurement variance increased systematically from 2.3% to 7.8% after 8-hour operator shifts, with particularly pronounced degradation during overnight testing periods when fatigue effects became most severe.",
        metrics: [
          { label: "Operating Pressure", value: "Up to 15,000 psi" },
          { label: "Temperature Range", value: "-40°C to 180°C" },
          { label: "Test Duration", value: "12-48 hours" },
          { label: "Manual Variance", value: "2.3% to 7.8%" }
        ],
        image: {
          src: "/lovable-uploads/86b80ba9-25f7-499c-95da-1e4d8d1511b8.png",
          alt: "Automated Valve Test Platform - Initial test setup with valve components, pressure transducers, and sensor wiring arranged in testing tray for high-pressure aerospace valve testing",
          position: "right"
        }
      },
      {
        type: "theoretical",
        title: "Theoretical Background",
        content: "Understanding the complex material science underlying valve failure mechanisms became crucial to developing an effective automated testing solution that could detect the subtle signatures of degradation processes. At elevated temperatures, valve components experience time-dependent plastic deformation following Norton's power law, where the creep rate depends exponentially on temperature and follows a power relationship with stress magnitude.\n\nFor Inconel 718 valve seats operating at 180°C under typical aerospace loading conditions, creep rates of 10⁻⁸ s⁻¹ can accumulate to significant deformation over extended test durations, with the exponential temperature dependence meaning that small temperature variations can dramatically affect creep behavior and thus failure timing.\n\nRepeated thermal cycling induces alternating stress cycles due to differential thermal expansion between dissimilar materials commonly found in valve assemblies. This leads to low-cycle fatigue crack initiation after 10³-10⁴ cycles, following the Coffin-Manson relationship.",
        image: {
          src: "/lovable-uploads/025a0eec-1e0e-43d8-ae59-443683cf3c02.png",
          alt: "Automated Valve Test Platform - Data acquisition software interface showing configuration menu for sensor setup, valve testing parameters, and real-time monitoring controls",
          position: "left"
        },
        equations: [
          {
            equation: "\\dot{\\varepsilon} = A\\left(\\frac{\\sigma}{E}\\right)^n \\exp\\left(-\\frac{Q}{RT}\\right)",
            variables: [
              { symbol: "ε̇", description: "creep rate (s⁻¹)" },
              { symbol: "A", description: "material constant" },
              { symbol: "σ", description: "applied stress (Pa)" },
              { symbol: "E", description: "elastic modulus (Pa)" },
              { symbol: "n", description: "stress exponent (3-8 for aerospace alloys)" },
              { symbol: "Q", description: "activation energy for creep (J/mol)" },
              { symbol: "R", description: "gas constant (8.314 J/mol·K)" },
              { symbol: "T", description: "absolute temperature (K)" }
            ]
          },
          {
            equation: "\\sigma_{th} = \\frac{E\\alpha\\Delta T}{1-\\nu}",
            variables: [
              { symbol: "σth", description: "thermal stress (Pa)" },
              { symbol: "E", description: "elastic modulus (Pa)" },
              { symbol: "α", description: "coefficient of thermal expansion (1/K)" },
              { symbol: "ΔT", description: "temperature change (K)" },
              { symbol: "ν", description: "Poisson's ratio" }
            ]
          },
          {
            equation: "\\frac{\\Delta\\varepsilon_p}{2} = \\varepsilon'_f(2N_f)^c",
            variables: [
              { symbol: "Δεp", description: "plastic strain amplitude" },
              { symbol: "ε'f", description: "fatigue ductility coefficient" },
              { symbol: "Nf", description: "cycles to failure" },
              { symbol: "c", description: "fatigue ductility exponent" }
            ]
          }
        ]
      },
      {
        type: "methodology",
        title: "Steps & Methodology",
        content: "My approach to solving this multifaceted challenge began with comprehensive hardware architecture design, carefully selecting components based on performance requirements, environmental compatibility, integration capabilities, and long-term reliability under the extreme conditions encountered in aerospace valve testing.\n\n**Hardware Selection Process:**\n\nThe Omega PX309 pressure transducers were chosen after extensive analysis across temperature ranges and competitive evaluation against alternatives from Honeywell, Kulite, and Kistler, featuring a measurement range of 0-15,000 psi with ±0.25% full scale accuracy, temperature coefficient of ±0.02% FS/°C, and response time under 1 ms for 90% full scale deflection.\n\n**Calibration Protocol:**\n\nThe calibration protocol represented a critical aspect requiring meticulous attention to traceability and uncertainty analysis. I utilized NIST-traceable Fluke 719Pro pressure calibrators with ±0.025% accuracy, deriving calibration equations through weighted least-squares regression that accounted for measurement uncertainties at each calibration point.\n\n**Temperature Measurement:**\n\nType-K thermocouples were selected for their wide operating range, standardized response characteristics defined by NIST standards, and proven reliability in aerospace applications. Cold junction compensation was achieved through integrated circuit temperature sensors (AD590) with ±0.5°C accuracy.",
        image: {
          src: "/lovable-uploads/7ca7784b-52b2-43fd-9fe0-970f628faa3a.png",
          alt: "Automated Valve Test Platform - Advanced test chamber setup with multiple valve assemblies, green PCB control boards, and comprehensive sensor network for environmental testing",
          position: "right"
        },
        standards: [
          "ASTM F1387 - Valve Testing Standards",
          "NASA-STD-5009 - Nondestructive Evaluation Requirements",
          "MIL-PRF-87257 - Military Performance Specifications",
          "NIST Standards - Calibration Traceability"
        ],
        equations: [
          {
            equation: "P_{actual} = a_0 + a_1 \\times V_{sensor} + a_2 \\times T_{ambient} + a_3 \\times V_{sensor} \\times T_{ambient} + a_4 \\times V_{sensor}^2",
            variables: [
              { symbol: "Pactual", description: "calibrated pressure reading (psi)" },
              { symbol: "Vsensor", description: "sensor voltage output (V)" },
              { symbol: "Tambient", description: "ambient temperature (°C)" },
              { symbol: "a0-a4", description: "calibration coefficients" }
            ]
          },
          {
            equation: "V(T) = \\sum_{i=0}^{n} a_i \\times T^i",
            variables: [
              { symbol: "V(T)", description: "thermocouple voltage (mV)" },
              { symbol: "T", description: "temperature (°C)" },
              { symbol: "ai", description: "NIST polynomial coefficients" }
            ]
          }
        ]
      },
      {
        type: "implementation",
        title: "Data & Results",
        content: "The data acquisition system selection involved extensive analysis of sampling rate requirements, channel count, resolution needs, and environmental compatibility. The National Instruments USB-6343 DAQ system provided 32 single-ended analog inputs with 16-bit resolution, maximum aggregate sampling rate of 500 kS/s, input voltage range of ±10V with programmable gain amplifiers, and built-in anti-aliasing filters.\n\n**Performance Achievements:**\n\n• Reduced measurement variance from 7.8% to 0.3%\n• Achieved continuous 24/7 operation capability\n• Implemented 1 Hz minimum sampling rate compliance\n• Eliminated operator fatigue-related errors\n• Achieved NIST-traceable calibration accuracy\n\n**System Capabilities:**\n\nThe 16-bit resolution provides theoretical measurement resolution of 1 part in 65,536, corresponding to 0.3 mV resolution over the ±10V input range, which translates to pressure resolution of approximately 0.08 psi when combined with sensor scaling factors. However, effective resolution is typically 2-3 bits less due to noise and environmental factors.",
        image: {
          src: "/lovable-uploads/2933f505-0bd4-4e2e-b1ae-b2f3601c9c3c.png",
          alt: "Automated Valve Test Platform - INFICON Go Skeleton Leak Fixture testing system with digital display showing troubleshoot mode, test loops, and real-time pressure monitoring for aerospace valve testing",
          position: "left"
        },
        metrics: [
          { label: "Measurement Accuracy", value: "±0.25% FS" },
          { label: "Temperature Coefficient", value: "±0.02% FS/°C" },
          { label: "Response Time", value: "<1 ms" },
          { label: "Resolution", value: "0.08 psi" },
          { label: "Sampling Rate", value: "500 kS/s" },
          { label: "Variance Improvement", value: "7.8% → 0.3%" }
        ],
        visual: {
          type: "chart",
          content: {
            type: "line",
            data: {
              labels: ["Manual (Start)", "Manual (8hr)", "Manual (16hr)", "Manual (24hr)", "Automated"],
              datasets: [{
                label: "Measurement Variance (%)",
                data: [2.3, 4.1, 6.8, 7.8, 0.3],
                borderColor: "hsl(var(--primary))",
                backgroundColor: "hsl(var(--primary) / 0.1)"
              }]
            }
          }
        }
      },
      {
        type: "results",
        title: "Mathematical Models & Equations",
        content: "The software architecture represented a sophisticated integration of real-time control, statistical analysis, safety monitoring, and data management systems that required careful attention to timing, reliability, and maintainability. I developed a modular Python framework employing object-oriented design principles with clear separation of concerns.",
        equations: [
          {
            equation: "k = A \\exp\\left(-\\frac{E_a}{RT}\\right)",
            variables: [
              { symbol: "k", description: "reaction rate constant (s⁻¹)" },
              { symbol: "A", description: "pre-exponential factor" },
              { symbol: "Ea", description: "activation energy (80-120 kJ/mol)" },
              { symbol: "R", description: "gas constant (8.314 J/mol·K)" },
              { symbol: "T", description: "absolute temperature (K)" }
            ]
          },
          {
            equation: "a_T = \\exp\\left[\\frac{E_a}{R}\\left(\\frac{1}{T} - \\frac{1}{T_{ref}}\\right)\\right]",
            variables: [
              { symbol: "aT", description: "shift factor for time-temperature superposition" },
              { symbol: "Ea", description: "activation energy (J/mol)" },
              { symbol: "T", description: "test temperature (K)" },
              { symbol: "Tref", description: "reference temperature (K)" }
            ]
          }
        ]
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
        <div className="container mx-auto px-4 py-24 text-center">
          <h1 className="text-4xl font-bold mb-4">Project Not Found</h1>
          <p className="text-muted-foreground mb-8">The project you're looking for doesn't exist.</p>
          <Button onClick={() => navigate('/projects')}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Projects
          </Button>
        </div>
        <Footer />
      </div>
    );
  }

  const renderSection = (section: any, index: number) => {
    const baseClasses = "py-16 border-b border-border/10 last:border-b-0";
    
    return (
      <section key={index} className={baseClasses}>
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold mb-8 text-primary">{section.title}</h2>
            
            {section.type === "overview" && (
              <div className={`grid gap-12 ${section.image ? 'lg:grid-cols-2' : 'grid-cols-1'} items-start`}>
                <div className={section.image?.position === "right" ? "order-1" : "order-2"}>
                  <div className="prose prose-lg max-w-none text-foreground">
                    {section.content.split('\n\n').map((paragraph: string, i: number) => (
                      <p key={i} className="mb-6 leading-relaxed">
                        {paragraph}
                      </p>
                    ))}
                  </div>
                  {section.metrics && (
                    <div className="grid grid-cols-2 gap-4 mt-8">
                      {section.metrics.map((metric: any, i: number) => (
                        <div key={i} className="bg-card p-4 rounded-lg border">
                          <div className="text-sm text-muted-foreground">{metric.label}</div>
                          <div className="text-xl font-bold text-primary">{metric.value}</div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
                {section.image && (
                  <div className={section.image.position === "right" ? "order-2" : "order-1"}>
                    <img 
                      src={section.image.src} 
                      alt={section.image.alt}
                      className="w-full rounded-lg shadow-lg"
                    />
                  </div>
                )}
              </div>
            )}

            {section.type === "theoretical" && (
              <div>
                <div className={`grid gap-12 ${section.image ? 'lg:grid-cols-2' : 'grid-cols-1'} items-start mb-12`}>
                  <div className={section.image?.position === "right" ? "order-1" : "order-2"}>
                    <div className="prose prose-lg max-w-none text-foreground">
                      {section.content.split('\n\n').map((paragraph: string, i: number) => (
                        <p key={i} className="mb-6 leading-relaxed">
                          {paragraph}
                        </p>
                      ))}
                    </div>
                  </div>
                  {section.image && (
                    <div className={section.image.position === "right" ? "order-2" : "order-1"}>
                      <img 
                        src={section.image.src} 
                        alt={section.image.alt}
                        className="w-full rounded-lg shadow-lg"
                      />
                    </div>
                  )}
                </div>
                {section.equations && (
                  <div className="space-y-8">
                    <h3 className="text-2xl font-bold mb-6">Mathematical Foundations</h3>
                    {section.equations.map((eq: any, i: number) => (
                      <MathEquation key={i} {...eq} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {section.type === "methodology" && (
              <div>
                <div className={`grid gap-12 ${section.image ? 'lg:grid-cols-2' : 'grid-cols-1'} items-start mb-12`}>
                  <div className={section.image?.position === "right" ? "order-1" : "order-2"}>
                    <div className="prose prose-lg max-w-none text-foreground">
                      {section.content.split('\n\n').map((paragraph: string, i: number) => (
                        <div key={i} className="mb-6">
                          {paragraph.startsWith('**') ? (
                            <h4 className="text-xl font-bold mb-3 text-primary">
                              {paragraph.replace(/\*\*/g, '')}
                            </h4>
                          ) : (
                            <p className="leading-relaxed">{paragraph}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                  {section.image && (
                    <div className={section.image.position === "right" ? "order-2" : "order-1"}>
                      <img 
                        src={section.image.src} 
                        alt={section.image.alt}
                        className="w-full rounded-lg shadow-lg"
                      />
                    </div>
                  )}
                </div>
                {section.standards && (
                  <div className="bg-card p-6 rounded-lg border">
                    <h4 className="text-lg font-bold mb-4 text-primary">Industry Standards Compliance</h4>
                    <ul className="space-y-2">
                      {section.standards.map((standard: string, i: number) => (
                        <li key={i} className="flex items-center text-sm">
                          <ChevronRight className="h-4 w-4 mr-2 text-primary" />
                          {standard}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {section.equations && (
                  <div className="space-y-8 mt-12">
                    <h3 className="text-2xl font-bold mb-6">Calibration Mathematics</h3>
                    {section.equations.map((eq: any, i: number) => (
                      <MathEquation key={i} {...eq} />
                    ))}
                  </div>
                )}
              </div>
            )}

            {section.type === "implementation" && (
              <div>
                <div className={`grid gap-12 ${section.image ? 'lg:grid-cols-2' : 'grid-cols-1'} items-start mb-12`}>
                  <div className={section.image?.position === "right" ? "order-1" : "order-2"}>
                    <div className="prose prose-lg max-w-none text-foreground">
                      {section.content.split('\n\n').map((paragraph: string, i: number) => (
                        <div key={i} className="mb-6">
                          {paragraph.startsWith('**') ? (
                            <h4 className="text-xl font-bold mb-3 text-primary">
                              {paragraph.replace(/\*\*/g, '')}
                            </h4>
                          ) : paragraph.startsWith('•') ? (
                            <ul className="list-disc list-inside space-y-2 ml-4">
                              {paragraph.split('\n').map((item: string, j: number) => (
                                <li key={j} className="leading-relaxed">{item.replace('• ', '')}</li>
                              ))}
                            </ul>
                          ) : (
                            <p className="leading-relaxed">{paragraph}</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                  {section.image && (
                    <div className={section.image.position === "right" ? "order-2" : "order-1"}>
                      <img 
                        src={section.image.src} 
                        alt={section.image.alt}
                        className="w-full rounded-lg shadow-lg"
                      />
                    </div>
                  )}
                </div>
                {section.metrics && (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-8">
                    {section.metrics.map((metric: any, i: number) => (
                      <div key={i} className="bg-card p-4 rounded-lg border text-center">
                        <div className="text-2xl font-bold text-primary mb-1">{metric.value}</div>
                        <div className="text-sm text-muted-foreground">{metric.label}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {section.type === "results" && (
              <div>
                <div className="prose prose-lg max-w-none text-foreground mb-12">
                  {section.content.split('\n\n').map((paragraph: string, i: number) => (
                    <p key={i} className="mb-6 leading-relaxed">
                      {paragraph}
                    </p>
                  ))}
                </div>
                {section.equations && (
                  <div className="space-y-8">
                    <h3 className="text-2xl font-bold mb-6">Key Equations</h3>
                    {section.equations.map((eq: any, i: number) => (
                      <MathEquation key={i} {...eq} />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>
    );
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="pt-24 pb-12 bg-gradient-to-b from-background to-background/95">
        <div className="container mx-auto px-4">
          <div className="max-w-6xl mx-auto">
            <Button 
              variant="ghost" 
              onClick={() => navigate('/projects')}
              className="mb-8 text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Projects
            </Button>
            
            <div className="grid lg:grid-cols-2 gap-12 items-center">
              <div>
                <Badge className="mb-4">{project.category}</Badge>
                <h1 className="text-4xl lg:text-5xl font-bold leading-tight mb-6">
                  {project.title}
                </h1>
                <p className="text-xl text-muted-foreground mb-6 leading-relaxed">
                  {project.subtitle}
                </p>
                
                <div className="flex flex-wrap gap-4 mb-6">
                  <div className="flex items-center text-sm text-muted-foreground">
                    <Calendar className="mr-2 h-4 w-4" />
                    {project.date}
                  </div>
                  <div className="flex items-center text-sm text-muted-foreground">
                    <User className="mr-2 h-4 w-4" />
                    {project.author}
                  </div>
                </div>
                
                <div className="flex flex-wrap gap-2">
                  {project.tags.map((tag: string) => (
                    <Badge key={tag} variant="secondary">{tag}</Badge>
                  ))}
                </div>
              </div>
              
              <div>
                <img 
                  src={project.hero} 
                  alt={project.title}
                  className="w-full rounded-lg shadow-2xl"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Content Sections */}
      {project.sections.map((section: any, index: number) => renderSection(section, index))}

      <ContactSection />
      <Footer />
    </div>
  );
};

export default ProjectDetail;