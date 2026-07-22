const VARIANTS = {
  primary: "ui-btn-primary",
  secondary: "ui-btn-secondary",
  danger: "ui-btn-danger",
};

export default function Button({
  variant = "primary",
  type = "button",
  className = "",
  children,
  ...props
}) {
  return (
    <button type={type} className={`${VARIANTS[variant] ?? VARIANTS.primary} ${className}`} {...props}>
      {children}
    </button>
  );
}
