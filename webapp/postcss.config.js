module.exports = {
  plugins: [
    require('autoprefixer'),
    require('postcss-preset-env')({
      stage: 1,
      features: {
        'custom-properties': false
      }
    }),
    require('cssnano')({
      preset: 'default'
    })
  ]
};
