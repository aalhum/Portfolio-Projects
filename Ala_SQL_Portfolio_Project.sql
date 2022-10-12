Select *
From PortfolioProject.dbo.CovidDeaths$
where continent IS NOT NULL
order by 3,4;

Select *
From PortfolioProject.dbo.CovidVaccinations$
order by 3,4;

-- Select data to use
Select Location, date, total_cases, new_cases, total_deaths, population
From PortfolioProject..CovidDeaths$
order by 1,2;

-- Looking at Total Cases vs Total Deaths
-- Shows the likelihood of dying if you contract covid in your country
Select Location, date, total_cases, total_deaths, (Total_deaths/total_cases)*100 as DeathPercentage
From PortfolioProject..CovidDeaths$
Where location like '%states%'
order by 1,2;

-- Looking at Total Cases vs Population
Select Location, date, total_cases, population, (total_cases/population)*100 as InfectedPercent
From PortfolioProject..CovidDeaths$
--Where location like '%states%'
order by 1,2;

-- Looking at countries with highest infection rates compared to population
Select Location, population, max(total_cases) as highestInfectionCount, max((total_cases/population))*100 as InfectedPercent
From PortfolioProject..CovidDeaths$
--Where location like '%states%'
Group by Location, Population
order by InfectedPercent desc;

-- Showing the countries with the highest death count per population
Select Location, Max(Total_Deaths) as TotalDeathCount
From PortfolioProject..CovidDeaths$
--Where location like '%states%'
where continent is not null
Group by Location
order by TotalDeathCount desc;

-- Let's break things down by continent
Select continent, Max(Total_Deaths) as TotalDeathCount
From PortfolioProject..CovidDeaths$
--Where location like '%states%'
where continent is not null
Group by continent
order by TotalDeathCount desc;

-- Global Numbers
Select sum(new_cases) as total_cases, SUM(cast(new_deaths as int)) as total_deaths, Sum(cast(new_deaths as int))/sum(new_cases)*100 as DeathPercentage--, total_deaths, (total_deaths/total_cases)*100 as InfectedPercent
From PortfolioProject..CovidDeaths$
--Where location like '%states%'
where continent is not null
--Group By date
order by 1,2;

-- Looking at Total Population vs Vaccinations
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(bigint,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/dea.Population)*100
From PortfolioProject.dbo.CovidVaccinations$ vac
Join PortfolioProject.dbo.CovidDeaths$ dea
	On dea.location = vac.location
	and dea.date = vac.date 
where dea.continent is not null
order by 1,2,3

-- use CTE
With PopvsVac (Continent, Location, Date, Population, New_Vaccinations, RollingPeopleVaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(bigint,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/dea.Population)*100
From PortfolioProject.dbo.CovidVaccinations$ vac
Join PortfolioProject.dbo.CovidDeaths$ dea
	On dea.location = vac.location
	and dea.date = vac.date 
where dea.continent is not null
--order by 1,2,3
)
Select *, (RollingPeopleVaccinated/Population)*100
From PopvsVac

-- use TEMP table
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
RollingPeopleVaccinated numeric
)
Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(bigint,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/dea.Population)*100
From PortfolioProject.dbo.CovidVaccinations$ vac
Join PortfolioProject.dbo.CovidDeaths$ dea
	On dea.location = vac.location
	and dea.date = vac.date 
where dea.continent is not null
--order by 1,2,3

Select *, (RollingPeopleVaccinated/Population)*100
From #PercentPopulationVaccinated

--Creating View to store data for later visualizations
Create View PercentPopulationVaccinated as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations
, SUM(CONVERT(bigint,vac.new_vaccinations)) OVER (Partition by dea.Location Order by dea.location, dea.date) as RollingPeopleVaccinated
--, (RollingPeopleVaccinated/dea.Population)*100
From PortfolioProject.dbo.CovidVaccinations$ vac
Join PortfolioProject.dbo.CovidDeaths$ dea
	On dea.location = vac.location
	and dea.date = vac.date 
where dea.continent is not null
--order by 2,3

Select *
From PercentPopulationVaccinated