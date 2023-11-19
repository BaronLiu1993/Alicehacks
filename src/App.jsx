import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [variables, setVariables] = useState([
    {Title: 'Web Design', Subtitle: 'Provide our customers with optimized user-friendly experience to increase the efficiency of digital products.', Link: 'See More!', id: 1},
    {Title: 'Web Design', Subtitle: 'Provide our customers with optimized user-friendly experience to increase the efficiency of digital products.', Link: 'See More!', id: 2},
    {Title: 'Web Design', Subtitle: 'Provide our customers with optimized user-friendly experience to increase the efficiency of digital products.', Link: 'See More!', id: 3},
    {Title: 'Web Design', Subtitle: 'Provide our customers with optimized user-friendly experience to increase the efficiency of digital products.', Link: 'See More!', id: 4},
    {Title: 'Web Design', Subtitle: 'Provide our customers with optimized user-friendly experience to increase the efficiency of digital products.', Link: 'See More!', id: 5},
    {Title: 'Web Design', Subtitle: 'Provide our customers with optimized user-friendly experience to increase the efficiency of digital products.', Link: 'See More!', id: 6},
  ])
  useEffect(() => {
    fetch('');
  })
  return (
    <>
      <div className = 'flex justify-between bg-[#fefefe]'>
        <div className = 'h-32 w-32 bg-black'>
          
        </div>
        <div className = 'flex space-x-[10rem] mr-52 mt-20'>
          <div>Services</div>
          <div>Works</div>
          <div>Blog</div>
          <div>Hire Me</div>
        </div>
      </div>

    <div className = 'm-48'>
      <div className = 'font-bold ml-2 text-xl'>
        Jessive Strootin
      </div>
      <div className = 'text-8xl font-bold'>
        Creative Thinker <br /> Minimalism Lover
      </div>
      <div className = 'text-gray-500 font-semi mt-20 text-3xl'>
        Hi I'm Jessica. I'm a UI/UX Designer. If you are looking for Designer <br /> to build your brands and grow your business. Let's shake <br /> hands with me 
      </div>
      <div className = 'flex mr-48 mt-20 space-x-[3rem]'>
        <a href = '#' className = 'bg-violet-300 p-6 rounded-lg border-violet-300 border-4 h-20 w-30 text-white font-bold hover:bg-white hover:text-black'>
          Hire Me
        </a>
        <a href = '#' className = 'bg-violet-300 p-6 rounded-lg border-violet-300 border-4 h-20 w-30 text-white font-bold hover:bg-white hover:text-black'>
          Read More
        </a>
      </div>
    </div>

    <div className = 'flex justify-center text-6xl mt-52'>
      <div className = 'font-bold center'>
      From beginning ideas to individual integrity, <br /> rich identity from the line on the paper to <br /> final projects
      </div>
    </div>

    <div>
      <div className = 'h-32 w-32 bg-black'>

      </div>
      <div className = ''>
        { variables.map((variable) => (
          <div key = {variable.id} >
            <div>
              { variable.Title }
            </div>
            <div>
              { variable.Subtitle }
            </div>
            <div>
              { variable.Link }
            </div>
          </div>
        )) }
      </div>
    </div>
    </>
  )
}

export default App
