declare global {
  namespace React {
    type ReactNode =
      | string
      | number
      | boolean
      | null
      | undefined
      | React.ReactElement
      | React.ReactFragment
      | React.ReactPortal;
  }
}