import { useEffect, useRef, useCallback, useMemo } from 'react';

export const useIntersectionObserver = (callback, options = {}) => {
  const observer = useRef(null);
  
  const defaultOptions = useMemo(() => ({
    rootMargin: '100px',
    threshold: 0.1,
    ...options
  }), [options]);

  useEffect(() => {
    observer.current = new IntersectionObserver(callback, defaultOptions);
    
    return () => {
      if (observer.current) {
        observer.current.disconnect();
      }
    };
  }, [callback, defaultOptions]);

  const observe = useCallback((element) => {
    if (observer.current && element) {
      observer.current.observe(element);
    }
  }, []);

  const unobserve = useCallback((element) => {
    if (observer.current && element) {
      observer.current.unobserve(element);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (observer.current) {
      observer.current.disconnect();
    }
  }, []);

  return { observe, unobserve, disconnect };
};
