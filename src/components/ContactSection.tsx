import { useEffect, useRef, useState } from "react";
import { Mail, Send, Linkedin, Twitter, Github } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

const ContactSection = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    message: ""
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    // Simulate form submission
    setTimeout(() => {
      toast({
        title: "Message sent!",
        description: "Thank you for reaching out. I'll get back to you soon.",
      });
      setFormData({ name: "", email: "", message: "" });
      setIsSubmitting(false);
    }, 1000);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }));
  };

  const socialLinks = [
    {
      name: "Email",
      icon: Mail,
      href: "mailto:Azarias.sime@ufl.edu",
      label: "Azarias.sime@ufl.edu"
    },
    {
      name: "LinkedIn",
      icon: Linkedin,
      href: "https://linkedin.com/in/azarias-sime-01823624a",
      label: "linkedin.com/in/azarias-sime-01823624a"
    },
    {
      name: "Phone",
      icon: Twitter,
      href: "tel:+14157456161",
      label: "(415) 745-6161"
    },
    {
      name: "GitHub",
      icon: Github,
      href: "https://github.com/azarias-sime",
      label: "github.com/azarias-sime"
    }
  ];

  return (
    <section
      id="contact"
      ref={sectionRef}
      className="py-24 md:py-32 section-padding"
    >
      <div className="container-max">
        {/* Section Header */}
        <div className={`text-center mb-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <h2 className="text-sm font-medium text-primary mb-4 tracking-wide uppercase">
            Contact
          </h2>
          <h3 className="text-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-8">
            Let's work together
          </h3>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            I'm always interested in new opportunities and collaborations. 
            Let's create something amazing together.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-16 max-w-6xl mx-auto">
          {/* Contact Form */}
          <div className={`${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-foreground mb-2">
                  Name
                </label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 bg-input border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-300 text-foreground placeholder-muted-foreground"
                  placeholder="Your full name"
                />
              </div>

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-foreground mb-2">
                  Email
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 bg-input border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-300 text-foreground placeholder-muted-foreground"
                  placeholder="your.email@example.com"
                />
              </div>

              <div>
                <label htmlFor="message" className="block text-sm font-medium text-foreground mb-2">
                  Message
                </label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  required
                  rows={6}
                  className="w-full px-4 py-3 bg-input border border-border rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent transition-all duration-300 text-foreground placeholder-muted-foreground resize-none"
                  placeholder="Tell me about your project or how we can collaborate..."
                />
              </div>

              <button
                type="submit"
                disabled={isSubmitting}
                className="btn-primary w-full inline-flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSubmitting ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Sending...
                  </>
                ) : (
                  <>
                    Send Message
                    <Send className="w-4 h-4" />
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Contact Information */}
          <div className={`${isVisible ? 'animate-fade-in' : 'opacity-0'}`} style={{ animationDelay: '0.2s' }}>
            <div className="bg-card border border-border rounded-2xl p-8">
              <h4 className="text-2xl font-bold mb-6 text-foreground">
                Get in touch
              </h4>
              
              <p className="text-muted-foreground mb-8 leading-relaxed">
                I'm currently seeking internships and full-time opportunities in aerospace engineering and manufacturing. 
                Whether you have a project in mind or want to discuss collaboration, I'd love to connect.
              </p>

              <div className="space-y-6">
                {socialLinks.map((social) => (
                  <a
                    key={social.name}
                    href={social.href}
                    target={social.name !== "Email" ? "_blank" : undefined}
                    rel={social.name !== "Email" ? "noopener noreferrer" : undefined}
                    className="flex items-center gap-4 p-4 rounded-lg hover:bg-muted transition-all duration-300 group"
                  >
                    <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center group-hover:bg-primary group-hover:text-white transition-all duration-300">
                      <social.icon className="w-5 h-5 text-primary group-hover:text-white" />
                    </div>
                    <div>
                      <h5 className="font-medium text-foreground">{social.name}</h5>
                      <p className="text-sm text-muted-foreground">{social.label}</p>
                    </div>
                  </a>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;