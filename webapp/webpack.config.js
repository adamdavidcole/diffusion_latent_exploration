const path = require('path');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  
  return {
    entry: {
      app: './frontend/static/js/app.js',
      controls: './frontend/static/js/controls.js'
    },
    output: {
      path: path.resolve(__dirname, 'frontend/static/dist'),
      filename: isProduction ? '[name].min.js' : '[name].js',
      clean: {
        keep: /\.css$/,  // Keep CSS files when cleaning
      }
    },
    devtool: isProduction ? false : 'source-map',
    module: {
      rules: [
        {
          test: /\.js$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: ['@babel/preset-env']
            }
          }
        }
      ]
    },
    optimization: {
      splitChunks: false,
      // Only minify in production
      minimize: isProduction
    }
  };
};
