const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="py-12 section-padding border-t border-border">
      <div className="container-max">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo and Copyright */}
          <div className="flex items-center gap-8">
            <div className="text-2xl font-bold text-foreground">
              Dinidu.
            </div>
            <p className="text-sm text-muted-foreground">
              © {currentYear} Dinidu Missaka. All rights reserved.
            </p>
          </div>

          {/* Contact Link */}
          <div className="flex items-center gap-6">
            <a
              href="mailto:dinidumissaka@gmail.com"
              className="text-muted-foreground hover:text-primary transition-colors duration-300"
            >
              dinidumissaka@gmail.com
            </a>
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-sm text-muted-foreground hover:text-primary transition-colors duration-300"
            >
              Back to top
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;